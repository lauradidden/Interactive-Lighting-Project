import time
from datetime import datetime, timezone
from astral.sun import sun
from astral import LocationInfo
import pyrealsense2 as rs
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import asyncio
from pyartnet import ArtNetNode
import random
from concurrent.futures import ThreadPoolExecutor
import signal
import logging
from threading import Lock

# Import GPIO for button handling
from gpiozero import Button

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize global variables
switch = "on"
sun_state = 'down'
kalman_lock = Lock()  # Lock for kalman filters
close_threshold = 50  # Threshold for considering two objects close
button_state = False  # State of the button
dmx_channels = {}  # Placeholder for DMX channels

# Global event loop reference
loop = None

# Button setup
button = Button(27)

def log_with_timestamp(message: str):
    logger.info(message)

async def send_zeros_to_dmx():
    """
    Sends zeros to all configured DMX channels.
    """
    zero_values = [0] * 9  # Assuming each channel has 9 attributes
    tasks = []
    for channel in dmx_channels.values():
        tasks.append(fade_to_color(channel, zero_values, 100))  # Short duration for immediate effect
    await asyncio.gather(*tasks)
    log_with_timestamp("Sent zeros to all DMX channels due to button press.")

def button_pressed():
    """
    Callback function when the button is pressed.
    Sets the button_state to True, indicating the button is pressed.
    Also, initiates sending zeros to the DMX channels.
    """
    global button_state
    button_state = True
    log_with_timestamp("Button pressed, pausing operations.")
    
    # Schedule the send_zeros_to_dmx task safely in the event loop
    if loop and loop.is_running():
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_zeros_to_dmx()))

def button_released():
    """
    Callback function when the button is released.
    Sets the button_state to False, indicating the button is released.
    """
    global button_state
    button_state = False
    log_with_timestamp("Button released, resuming operations.")

# Assign button press and release callbacks
button.when_pressed = button_pressed
button.when_released = button_released

def setup_location_info() -> LocationInfo:
    return LocationInfo("Amsterdam", "Netherlands")

def update_sun_state(city: LocationInfo) -> str:
    now = datetime.now(timezone.utc)
    s = sun(city.observer, date=now.date())
    sunrise = s['sunrise']
    sunset = s['sunset']
    return 'up' if sunrise <= now <= sunset else 'down'

def create_kalman_filter() -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R = np.array([[5, 0], [0, 5]])
    kf.Q = np.eye(4)
    return kf

async def fade_to_color(channel, colors, duration):
    channel.add_fade(colors, duration)
    await channel

def generate_northern_lights_color() -> list:
    r = random.randint(0, 50)    
    g = random.randint(0, 100)   
    b = random.randint(200, 255) 
    w = random.randint(50, 150)
    zoom = random.randint(50, 150)
    return [255, r, g, b, w, 0, zoom, 0, 0]

def generate_warm_sun_cloud_color() -> list:
    d = random.randint(200, 255)  
    r = random.randint(200, 255)  
    g = random.randint(200, 255)
    b = random.randint(0, 50)    
    w = random.randint(255, 255)  
    zoom = random.randint(50, 150)
    return [d, r, g, b, w, 0, zoom, 0, 0]

def euclidean_distance(p1, p2) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def process_frame(color_image, bg_subtractor_rgb, kalman_filters):
    fg_mask_rgb = bg_subtractor_rgb.apply(color_image)
    _, fg_mask_rgb = cv2.threshold(fg_mask_rgb, 254, 255, cv2.THRESH_BINARY)
    fg_mask_rgb = cv2.erode(fg_mask_rgb, None, iterations=2)
    fg_mask_rgb = cv2.dilate(fg_mask_rgb, None, iterations=2)

    contours_rgb, _ = cv2.findContours(fg_mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours_rgb:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            valid_contours.append((x, y, w, h))

    new_kalman_filters = []
    for (x, y, w, h) in valid_contours:
        cx = x + w / 2
        cy = y + h / 2
        kf = create_kalman_filter()
        kf.x[:2] = [cx, cy]
        kf.update(np.array([cx, cy]))
        new_kalman_filters.append(kf)

    return new_kalman_filters

def check_trigger_and_draw_lines(centroids, close_threshold) -> bool:
    triggered = False
    for i, centroid1 in enumerate(centroids):
        for centroid2 in centroids[i + 1:]:
            if euclidean_distance(centroid1, centroid2) < close_threshold:
                triggered = True
                centroid1 = tuple(map(int, centroid1))
                centroid2 = tuple(map(int, centroid2))
    return triggered

async def combined_sun_loop(node, universe, pipeline, bg_subtractor_rgb, executor, city):
    global dmx_channels
    channel1 = universe.add_channel(start=1, width=9)
    channel2 = universe.add_channel(start=10, width=9)
    channel3 = universe.add_channel(start=19, width=9)

    # Store channels for easy access during zero transmission
    dmx_channels = {
        'channel1': channel1,
        'channel2': channel2,
        'channel3': channel3
    }

    kalman_filters = []

    while switch == 'on':
        if button_state:  # Pause operations if the button is pressed
            await send_zeros_to_dmx()  # Ensure zeros are sent during the pause
            await asyncio.sleep(0.1)
            continue  # Skip the rest of the loop and wait for the button to be released

        duration = random.randint(1000, 3000)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            await asyncio.sleep(0.1)
            continue

        color_image = np.asanyarray(color_frame.get_data())

        with kalman_lock:
            kalman_filters = await asyncio.get_event_loop().run_in_executor(
                executor, process_frame, color_image, bg_subtractor_rgb, kalman_filters
            )

        area_1 = (0, 213)
        area_2 = (214, 426)
        area_3 = (427, 640)

        area1_triggered = area2_triggered = area3_triggered = False

        centroids_area1 = []
        centroids_area2 = []
        centroids_area3 = []

        for kf in kalman_filters:
            tracked_x, tracked_y = kf.x[:2]

            if area_1[0] <= tracked_x <= area_1[1]:
                centroids_area1.append((int(tracked_x), int(tracked_y)))
            elif area_2[0] <= tracked_x <= area_2[1]:
                centroids_area2.append((int(tracked_x), int(tracked_y)))
            elif area_3[0] <= tracked_x <= area_3[1]:
                centroids_area3.append((int(tracked_x), int(tracked_y)))

        area1_triggered = check_trigger_and_draw_lines(centroids_area1, close_threshold)
        area2_triggered = check_trigger_and_draw_lines(centroids_area2, close_threshold)
        area3_triggered = check_trigger_and_draw_lines(centroids_area3, close_threshold)

        global sun_state
        sun_state = update_sun_state(city)

        if sun_state == 'up':
            colors1 = generate_warm_sun_cloud_color()
            colors2 = generate_warm_sun_cloud_color()
            colors3 = generate_warm_sun_cloud_color()
        else:
            colors1 = generate_northern_lights_color()
            colors2 = generate_northern_lights_color()
            colors3 = generate_northern_lights_color()

        if area1_triggered:
            log_with_timestamp(f"Area 1 {sun_state} triggered")
            colors1 = [255, 255, 0, 127, 255, 0, 40, 0, 0] if sun_state == 'up' else [255, 255, 92, 0, 255, 0, 40, 0, 0]
        if area2_triggered:
            log_with_timestamp(f"Area 2 {sun_state} triggered")
            colors2 = [255, 255, 0, 127, 255, 0, 40, 0, 0] if sun_state == 'up' else [255, 255, 92, 0, 255, 0, 40, 0, 0]
        if area3_triggered:
            log_with_timestamp(f"Area 3 {sun_state} triggered")
            colors3 = [255, 255, 0, 127, 255, 0, 40, 0, 0] if sun_state == 'up' else [255, 255, 92, 0, 255, 0, 40, 0, 0]

        log_with_timestamp(f"Sending DMX values: {colors1}, {colors2}, {colors3}")

        await asyncio.gather(
            fade_to_color(channel1, colors1, duration),
            fade_to_color(channel2, colors2, duration),
            fade_to_color(channel3, colors3, duration)
        )

        await asyncio.sleep(0.1)

async def main():
    global loop
    loop = asyncio.get_event_loop()  # Get the current event loop
    city = setup_location_info()
    node = ArtNetNode('192.168.0.84', 6454)
    universe = node.add_universe(0)

    pipeline, bg_subtractor_rgb = initialize_realsense_and_subtractor()

    with ThreadPoolExecutor() as executor:
        try:
            while switch == 'on':
                await combined_sun_loop(node, universe, pipeline, bg_subtractor_rgb, executor, city)
        finally:
            cleanup_resources(pipeline)

def initialize_realsense_and_subtractor():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    bg_subtractor_rgb = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    return pipeline, bg_subtractor_rgb

def cleanup_resources(pipeline):
    if pipeline:
        pipeline.stop()
    cv2.destroyAllWindows()

def handle_exit_signal():
    global switch
    switch = "off"
    log_with_timestamp("Received exit signal, shutting down...")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()

        signal.signal(signal.SIGINT, lambda s, f: handle_exit_signal())
        signal.signal(signal.SIGTERM, lambda s, f: handle_exit_signal())

        loop.run_until_complete(main())
    except Exception as e:
        log_with_timestamp(f"An error occurred: {e}")
        cleanup_resources(None)
