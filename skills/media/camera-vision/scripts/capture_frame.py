#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def capture_with_opencv(device_path, output_path, burst=1, interval=0.5):
    """Try to capture frame(s) using OpenCV."""
    try:
        import cv2
        if device_path.startswith("/dev/video"):
            try:
                device_index = int(device_path.replace("/dev/video", ""))
            except ValueError:
                device_index = device_path
        else:
            device_index = device_path

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return False

        # Allow camera to warm up
        time.sleep(0.5)
        
        captured_files = []
        for i in range(burst):
            # Capture multiple frames to discard the first few (often dark)
            for _ in range(5):
                ret, frame = cap.read()
            
            if ret:
                current_output = output_path
                if burst > 1:
                    ext = output_path.suffix
                    current_output = output_path.parent / f"{output_path.stem}_{i}{ext}"
                
                cv2.imwrite(str(current_output), frame)
                captured_files.append(str(current_output))
                if i < burst - 1:
                    time.sleep(interval)
            else:
                break
        
        cap.release()
        if captured_files:
            print(f"Captured {len(captured_files)} frames: {', '.join(captured_files)}")
            return True
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Capture frame(s) from the camera.")
    parser.add_argument("--device", default="/dev/video1", help="Camera device path (default: /dev/video1)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--burst", type=int, default=1, help="Number of frames to capture for sequence analysis")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between burst frames")
    
    args = parser.parse_args()
    
    if args.output:
        output_path = Path(args.output)
    else:
        temp_dir = Path("./temp_vision_images")
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / "camera_capture.jpg"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try requested device first
    if capture_with_opencv(args.device, output_path, args.burst, args.interval):
        sys.exit(0)
    
    # Fallback to other common devices
    for dev in ["/dev/video0", "/dev/video2"]:
        if dev == args.device: continue
        if capture_with_opencv(dev, output_path, args.burst, args.interval):
            sys.exit(0)
    
    print("Failed to capture image.")
    sys.exit(1)

if __name__ == "__main__":
    main()
