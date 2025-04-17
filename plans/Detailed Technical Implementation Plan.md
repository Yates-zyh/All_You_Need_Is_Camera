# Detailed Technical Implementation Plan (Python/PyTorch)

This document provides a step-by-step technical guide for implementing the "All You Need Is Camera" interaction framework using Python and the PyTorch ecosystem, focusing on YOLOv8-Pose.

## 1. Environment Setup

* **Python:** Version 3.8+ recommended.
* **Package Manager:** `pip` or `conda`.
* **Core Libraries:**
    * `torch`: For PyTorch models and tensor operations. Install version compatible with your CUDA version if using GPU acceleration.
    * `ultralytics`: The official package for YOLO models. (`pip install ultralytics`)
    * `opencv-python`: For camera access and image processing (`cv2`). (`pip install opencv-python`)
    * `numpy`: For numerical operations. (`pip install numpy`)
* **Game/UI Library (Choose based on target game):**
    * `pygame`: Good for rapid prototyping of 2D games. (`pip install pygame`)
    * *(Alternatives: Kivy, Pyglet, or integrating with web frameworks via backend)*
* **(Optional) Tracking Libraries:**
    * `filterpy`: Contains implementations for Kalman Filters (often used in SORT).
    * `scipy`: For spatial distance calculations, potentially needed for SORT/similarity metrics. (`pip install scipy`)
* **(Optional) Performance/Deployment:**
    * `onnx`, `onnxruntime-gpu` / `onnxruntime`: For exporting models and running with ONNX Runtime.
    * `tensorrt` (via NVIDIA containers or manual installation): For optimizing models on NVIDIA GPUs.

## 2. Core Framework Implementation (Python/PyTorch)

This involves creating a reusable Python module/class that handles camera input and pose estimation.

```python
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

# --- Configuration ---
MODEL_PATH = 'yolov8n-pose.pt' # Or yolov8s-pose.pt, etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# --- Optional: Tracking Configuration ---
TRACK_BUFFER_SIZE = 10 # Store last N keypoints for trajectory

class PoseFramework:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        """Initializes the YOLOv8-Pose model."""
        self.device = device
        self.model = YOLO(model_path)
        # Ensure model is on the correct device
        # Note: Ultralytics YOLO might handle device placement automatically,
        # but explicit placement can be safer. Check their docs.
        # self.model.to(self.device) # May not be needed directly with YOLO class

        # --- Optional: Tracking State ---
        self.tracked_poses = {} # Dictionary to store state per tracked ID
                                # Example: {track_id: {'bbox': [], 'keypoints': deque(), 'last_seen': timestamp}}

    def process_frame(self, frame):
        """Processes a single frame to detect and track poses."""
        # --- 1. YOLOv8-Pose Inference ---
        # Use torch.no_grad() for efficiency during inference
        with torch.no_grad():
            # Perform inference. Check ultralytics docs for exact parameters (e.g., verbose=False)
            results = self.model.predict(frame, device=self.device, verbose=False) # Can also use model.track for integrated tracking

        # --- 2. Parse Results ---
        # Results[0] typically contains detections for the first image in the batch
        detected_persons = []
        if results and results[0].keypoints and results[0].boxes:
             keypoints_tensor = results[0].keypoints.data # Tensor of keypoints (N, K, 2/3) [N persons, K keypoints, xy/xy_conf]
             boxes_tensor = results[0].boxes.data       # Tensor of bounding boxes (N, 6) [xyxy, conf, class_id]

             # Ensure tensors are on CPU for numpy conversion if needed
             keypoints_np = keypoints_tensor.cpu().numpy()
             boxes_np = boxes_tensor.cpu().numpy()

             for i in range(len(boxes_np)):
                 person_data = {
                     'id': None, # Will be filled by tracking later if enabled
                     'bbox': boxes_np[i][:4], # xyxy
                     'confidence': boxes_np[i][4],
                     'keypoints_xy': keypoints_np[i][:, :2], # (K, 2) array of xy coordinates
                     'keypoints_conf': keypoints_np[i][:, 2] if keypoints_np.shape[2] > 2 else np.ones(keypoints_np.shape[1]), # Confidence if available
                     'trajectory': None # Will be filled by tracking
                 }
                 # --- Filter based on confidence (example threshold) ---
                 if person_data['confidence'] > 0.5:
                     detected_persons.append(person_data)

        # --- 3. Optional: Tracking (Simplified Example using dictionary) ---
        # A proper tracker (SORT, DeepSORT) would be more robust
        # This is a placeholder illustrating state maintenance
        current_time = time.time()
        updated_tracked_poses = {}
        # (Here you would implement matching logic: e.g., IoU between current detections
        # and last known positions of tracked objects)
        # For now, let's just update trajectories for the first detected person as an example
        if detected_persons:
            # Assume person 0 is being tracked with ID 1 for simplicity
            track_id = 1
            if track_id not in self.tracked_poses:
                self.tracked_poses[track_id] = {'keypoints_history': deque(maxlen=TRACK_BUFFER_SIZE)}
            self.tracked_poses[track_id]['keypoints_history'].append(detected_persons[0]['keypoints_xy'])
            detected_persons[0]['id'] = track_id
            detected_persons[0]['trajectory'] = list(self.tracked_poses[track_id]['keypoints_history'])
            # Keep track of updated poses
            # updated_tracked_poses[track_id] = self.tracked_poses[track_id]

        # self.tracked_poses = updated_tracked_poses # Update tracked state

        # --- 4. Prepare Output API ---
        framework_output = {
            'timestamp': current_time,
            'persons': detected_persons # List of dictionaries, each describing a detected person
        }
        return framework_output

# --- Example Usage ---
# cap = cv2.VideoCapture(0) # Use 0 for default webcam
# framework = PoseFramework()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     processed_data = framework.process_frame(frame)
#
#     # --- Visualization (Example) ---
#     vis_frame = frame.copy()
#     for person in processed_data['persons']:
#         # Draw bounding box
#         x1, y1, x2, y2 = map(int, person['bbox'])
#         cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # Draw keypoints
#         for i, (x, y) in enumerate(person['keypoints_xy']):
#             if person['keypoints_conf'][i] > 0.5: # Draw only confident keypoints
#                  cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
#         # Draw trajectory for wrist (keypoint index 9 often corresponds to right wrist)
#         if person['trajectory'] and len(person['trajectory']) > 1:
#              wrist_idx = 9
#              pts = np.array([hist[wrist_idx] for hist in person['trajectory'] if hist[wrist_idx][0] > 0], dtype=np.int32) # Get history for right wrist
#              if len(pts) > 1:
#                   cv2.polylines(vis_frame, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
#
#     cv2.imshow('Pose Framework Output', vis_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()