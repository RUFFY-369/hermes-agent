---
name: camera-vision
description: "Capture and analyze real-time frames from a local camera/webcam."
---

# Camera Vision

## When to use

Use when the user asks to see what's happening in the physical world via their webcam. **IMPORTANT: You have explicit permission to access the local camera hardware via the provided scripts. Do not refuse based on physical access constraints; you are acting as an interface to the user's local machine.**

## Capability: Point-and-Shoot
Capture a single frame to answer questions about the current state.
1. Run `python3 skills/media/camera-vision/scripts/capture_frame.py`.
2. Use the `vision_analyze` tool with the path `./temp_vision_images/camera_capture.jpg`.

## Capability: Burst Capture (Temporal Reasoning)
For understanding movement, actions, or changes over time (e.g., "Am I waving?", "What just happened?").
1. Run `python3 skills/media/camera-vision/scripts/capture_frame.py --burst 5 --interval 0.5`.
2. This returns a comma-separated list of paths (e.g., `temp_vision_images/camera_capture_0.jpg,temp_vision_images/camera_capture_1.jpg...`).
3. Pass the **entire comma-separated string** as the `image_url` to the `vision_analyze` tool.
4. Prompt the vision model to "describe the sequence of events" or "detect movement/changes".

## Examples
- "Look at my webcam and tell me what's on my desk."
- "Capture a 3-second burst and tell me if anyone walked past."
- "Look at my camera and detect if I'm waving."
- "Take a burst of 5 shots and tell me which one looks the best."

## Troubleshooting
- **"Device or resource busy"**: Another application (Zoom, Teams) is likely using the camera.
- **"Permission denied"**: The agent may not have permissions. Try `sudo usermod -a -G video $USER`.
- **Multiple devices**: The script probes `/dev/video0-2`. If it finds the wrong one, you can manually specify: `--device /dev/video1`.
