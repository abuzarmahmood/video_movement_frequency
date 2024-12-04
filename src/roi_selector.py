import cv2
import json
import os
import argparse
from pathlib import Path
import numpy as np
from camera_process import get_capture, artifact_dir

class ROISelector:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.roi = []
        self.drawing = False
        self.cap = get_capture(device_id)
        self.window_name = "Draw ROI - Press SPACE when done, ESC to cancel"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            frame_copy = self.current_frame.copy()
            cv2.rectangle(frame_copy, self.roi[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi.append((x, y))
            cv2.rectangle(self.current_frame, self.roi[0], self.roi[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_frame)

    def select_roi(self):
        """Let user draw ROI on camera feed"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            cv2.imshow(self.window_name, self.current_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.roi = []
                break
            elif key == 32:  # SPACE
                if len(self.roi) == 2:
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        
        if len(self.roi) == 2:
            # Convert to x,y,w,h format
            x1, y1 = self.roi[0]
            x2, y2 = self.roi[1]
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            return [x, y, w, h]
        return None

def save_roi(roi_coords, device_id):
    """Save ROI coordinates to JSON file"""
    roi_file = os.path.join(artifact_dir, f"roi_device_{device_id}.json")
    with open(roi_file, 'w') as f:
        json.dump({
            'device_id': device_id,
            'roi': roi_coords
        }, f, indent=2)
    return roi_file

def main():
    parser = argparse.ArgumentParser(description='Select ROI for camera')
    parser.add_argument('--device-id', type=int, default=0,
                      help='Camera device index (default: 0)')
    args = parser.parse_args()

    selector = ROISelector(args.device_id)
    roi = selector.select_roi()
    
    if roi:
        roi_file = save_roi(roi, args.device_id)
        print(f"ROI saved to {roi_file}")
        print(f"ROI coordinates (x,y,w,h): {roi}")
    else:
        print("ROI selection cancelled")

if __name__ == '__main__':
    main()
