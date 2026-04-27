#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def capture_with_opencv(device_path, output_path):
    """Try to capture a frame using OpenCV."""
    try:
        import cv2
        # On Linux, device_path is usually /dev/videoN
        # cv2.VideoCapture takes an integer index or a path string
        # If it's a string like "/dev/video0", it might need specific backend
        
        # Try to extract the index from /dev/videoN
        if device_path.startswith("/dev/video"):
            try:
                device_index = int(device_path.replace("/dev/video", ""))
            except ValueError:
                device_index = device_path
        else:
            device_index = device_path

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            print(f"OpenCV: Could not open device {device_path}")
            return False

        # Allow camera to warm up
        time.sleep(0.5)
        
        # Capture multiple frames to discard the first few (often dark)
        for _ in range(5):
            ret, frame = cap.read()
            
        if ret:
            cv2.imwrite(str(output_path), frame)
            cap.release()
            print(f"OpenCV: Successfully captured frame to {output_path}")
            return True
        else:
            print("OpenCV: Failed to read frame")
            cap.release()
            return False
    except ImportError:
        print("OpenCV: cv2 module not found")
        return False
    except Exception as e:
        print(f"OpenCV: Error: {e}")
        return False

def capture_with_ffmpeg(device_path, output_path):
    """Try to capture a frame using ffmpeg."""
    try:
        # Command: ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video0 -frames:v 1 out.jpg
        # We use -y to overwrite and -t 1 to avoid hanging if the device is busy
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "v4l2",
            "-i", device_path,
            "-frames:v", "1",
            "-q:v", "2",  # High quality
            str(output_path)
        ]
        
        # Suppress ffmpeg banner and info logs
        result = subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0 and output_path.exists():
            print(f"ffmpeg: Successfully captured frame to {output_path}")
            return True
        else:
            error = result.stderr.decode('utf-8', errors='ignore')
            print(f"ffmpeg: Failed with return code {result.returncode}")
            if "Device or resource busy" in error:
                print("ffmpeg: Device is busy. Ensure no other application is using the camera.")
            return False
    except subprocess.TimeoutExpired:
        print("ffmpeg: Command timed out")
        return False
    except Exception as e:
        print(f"ffmpeg: Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Capture a frame or burst from the camera.")
    parser.add_argument("--device", default="/dev/video1", help="Camera device path (default: /dev/video1)")
    parser.add_argument("--output", help="Output file path pattern (default: temp_vision_images/camera_capture.jpg)")
    parser.add_argument("--burst", type=int, default=1, help="Number of frames to capture (default: 1)")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between frames in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    device_path = args.device
    temp_dir = Path("./temp_vision_images")
    temp_dir.mkdir(exist_ok=True)

    # If default device is requested, try to find a working one
    devices_to_try = [device_path]
    if device_path == "/dev/video1":
        devices_to_try.extend(["/dev/video0", "/dev/video2"])

    working_device = None
    for dev in devices_to_try:
        # Check if device exists and is accessible
        if os.path.exists(dev):
            working_device = dev
            break
    
    if not working_device:
        print(f"Error: No camera devices found (tried {', '.join(devices_to_try)})")
        sys.exit(1)

    captured_paths = []
    
    for i in range(args.burst):
        if args.output:
            if args.burst > 1:
                # Add index to output name if burst
                p = Path(args.output)
                output_path = p.parent / f"{p.stem}_{i}{p.suffix}"
            else:
                output_path = Path(args.output)
        else:
            suffix = f"_{i}" if args.burst > 1 else ""
            output_path = temp_dir / f"camera_capture{suffix}.jpg"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Capture from working device
        print(f"[{i+1}/{args.burst}] Capturing from {working_device}...")
        
        success = capture_with_opencv(working_device, output_path)
        if not success:
            print(f"OpenCV failed for {working_device}, trying ffmpeg fallback...")
            success = capture_with_ffmpeg(working_device, output_path)
        
        if success:
            captured_paths.append(str(output_path))
        else:
            print(f"Failed to capture frame {i+1}")
            break
            
        if i < args.burst - 1:
            time.sleep(args.interval)

    if captured_paths:
        # Output the list of paths for the agent to use
        print("\nSUCCESS: Captured images:")
        print(",".join(captured_paths))
        sys.exit(0)
    else:
        print("Failed to capture any images.")
        sys.exit(1)

if __name__ == "__main__":
    main()
