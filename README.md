# 💡 Real-Time Object Detection & Smart Lighting

A real-time lighting control system that uses computer vision and environmental context to dynamically control DMX lighting. Designed for interactive spaces where lighting reacts to human presence and time of day.

## Features

- Real-time object tracking using Intel RealSense RGB camera
- Adaptive lighting based on sun position (via Astral)
- Motion-triggered light responses across 3 screen zones
- GPIO-connected button for immediate light override (DMX blackout)
- DMX output over Art-Net protocol

<img src="https://github.com/user-attachments/assets/2de7d450-a7d3-40a2-9bd2-daca3a457700" width="500"/>

## Hardware Requirements

- **Intel RealSense D435 or D415** (RGB depth camera; only RGB stream used)
- **Raspberry Pi 5**
- **Art-Net compatible DMX controller**
- **Programmable DMX lighting fixtures**
- **Momentary push-button** connected to **GPIO pin 27**
