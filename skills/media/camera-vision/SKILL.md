---
name: camera-vision
description: "See through the user's camera in real-time or sequence."
---

# Camera Vision

## When to use

Use when the user asks "What do you see?", "What am I doing?", or "Watch this".

## Workflow

1.  **Capture Sequence**: Run the capture script with `--burst 5` to get a 3-second sequence of frames.
    ```bash
    python3 skills/media/camera-vision/scripts/capture_frame.py --device /dev/video1 --burst 5
    ```
2.  **Analyze**: Send all 5 images to `vision_analyze` and ask the model to describe the **action or movement** across the sequence.

## Example

- User: "Watch me and tell me what I'm doing."
- Hermes: (Runs capture with burst, analyzes sequence, describes the motion)
