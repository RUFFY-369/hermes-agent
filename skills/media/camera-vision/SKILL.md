---
name: camera-vision
description: "Capture and analyze real-time frames from a local camera/webcam."
---

# Camera Vision

## When to use

Use when the user asks to see what's happening in the physical world via their webcam, asks "What do you see on my camera?", "Who is behind me?", or requests real-time monitoring of their environment. 

## Workflow

1.  **Capture**: Run the capture script to take a snapshot from the camera.
    ```bash
    # Using the default device (/dev/video0)
    python3 skills/media/camera-vision/scripts/capture_frame.py
    
    # Or specify a different device if /dev/video0 fails
    python3 skills/media/camera-vision/scripts/capture_frame.py --device /dev/video1
    ```
    The script saves the image to `./temp_vision_images/camera_capture.jpg` by default.

2.  **Analyze**: Use the built-in `vision_analyze` tool to process the captured image.
    *   **image_url**: `./temp_vision_images/camera_capture.jpg`
    *   **question**: The user's specific query about the camera feed.

3.  **Repeat (Optional)**: If the user asks to "keep watching" or for "updates", repeat the process at appropriate intervals (e.g., every 10-30 seconds).

## Troubleshooting

- **"Device or resource busy"**: Another application (like Zoom, Teams, or a browser) is likely using the camera. Ask the user to close other camera apps.
- **"Permission denied"**: The agent may not have permissions to `/dev/video0`. Suggest the user adds themselves to the `video` group: `sudo usermod -a -G video $USER`.
- **Black frames**: Some cameras need longer to warm up. The script includes a 0.5s delay, but if frames are still dark, suggest the user checks lighting or camera covers.
- **Multiple devices**: If `/dev/video0` is not the correct camera, try `/dev/video1` or `/dev/video2`.
