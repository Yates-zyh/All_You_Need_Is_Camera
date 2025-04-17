"""
Core pose detection and tracking framework using YOLOv8-Pose.
"""
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PoseFramework:
    """YOLOv8-Pose based framework for human pose detection and tracking."""

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        track_buffer_size: int = 10,
    ):
        """
        Initialize the YOLOv8-Pose framework.

        Args:
            model_path: Path to the YOLOv8 pose model weights.
            device: Device to run the model on ('cuda' or 'cpu'). If None, will use cuda if available.
            confidence_threshold: Minimum detection confidence to consider.
            track_buffer_size: Number of past frames to track for trajectory analysis.
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load YOLOv8 model
        self.model = YOLO(model_path)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.track_buffer_size = track_buffer_size
        
        # Tracking state
        self.tracked_poses = {}  # Dictionary to store state per tracked ID
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame to detect and track poses.
        
        Args:
            frame: Input image frame as numpy array (BGR format from OpenCV).
            
        Returns:
            Dictionary containing detected persons with their poses and tracking info.
        """
        # Record timestamp
        current_time = time.time()
        
        # Run YOLOv8-Pose inference
        with torch.no_grad():
            results = self.model.predict(
                frame,
                device=self.device,
                verbose=False,
                conf=self.confidence_threshold
            )
        
        # Parse results
        detected_persons = []
        if results and len(results) > 0:
            # Check if keypoints and boxes are available
            if hasattr(results[0], 'keypoints') and hasattr(results[0], 'boxes'):
                keypoints_tensor = results[0].keypoints.data
                boxes_tensor = results[0].boxes.data
                
                # Convert to numpy for processing
                keypoints_np = keypoints_tensor.cpu().numpy()
                boxes_np = boxes_tensor.cpu().numpy()
                
                # Process each detected person
                for i in range(len(boxes_np)):
                    person_data = {
                        'id': None,  # Will be filled by tracking
                        'bbox': boxes_np[i][:4].tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(boxes_np[i][4]),
                        'keypoints_xy': keypoints_np[i, :, :2].tolist(),  # [N, 2] array of [x,y]
                        'keypoints_conf': keypoints_np[i, :, 2].tolist() if keypoints_np.shape[2] > 2 else [1.0] * keypoints_np.shape[1],
                        'trajectory': None  # Will be filled by tracking
                    }
                    
                    # Add to detected persons list if confidence is sufficient
                    if person_data['confidence'] > self.confidence_threshold:
                        detected_persons.append(person_data)
        
        # Simple tracking (Just for trajectory tracking in MVP)
        # In a more complete implementation, we would use a proper tracker like SORT/DeepSORT
        updated_tracked_poses = {}
        
        # For each detected person, try to match with existing tracks
        for i, person in enumerate(detected_persons):
            # For simplicity in MVP, we track the first person as ID 1
            # In a real application, we'd match based on IoU or feature similarity
            track_id = 1 if i == 0 else i + 1
            
            # Initialize track if new
            if track_id not in self.tracked_poses:
                self.tracked_poses[track_id] = {
                    'keypoints_history': deque(maxlen=self.track_buffer_size)
                }
            
            # Update track
            self.tracked_poses[track_id]['keypoints_history'].append(person['keypoints_xy'])
            self.tracked_poses[track_id]['last_seen'] = current_time
            
            # Add tracking info to person data
            person['id'] = track_id
            person['trajectory'] = list(self.tracked_poses[track_id]['keypoints_history'])
            
            # Keep track updated
            updated_tracked_poses[track_id] = self.tracked_poses[track_id]
        
        # Update tracked poses
        self.tracked_poses = updated_tracked_poses
        
        # Prepare output
        framework_output = {
            'timestamp': current_time,
            'persons': detected_persons
        }
        
        return framework_output
    
    def draw_results(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw detection and tracking results on a frame.
        
        Args:
            frame: Input image frame.
            pose_data: Pose data from process_frame().
            
        Returns:
            Frame with visualizations.
        """
        vis_frame = frame.copy()
        
        for person in pose_data['persons']:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, person['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            if person['id'] is not None:
                cv2.putText(
                    vis_frame, 
                    f"ID: {person['id']}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            # Draw keypoints
            keypoints = person['keypoints_xy']
            confidences = person['keypoints_conf']
            
            # COCO keypoint connections (simplification)
            connections = [
                (5, 7), (7, 9),     # Left arm
                (6, 8), (8, 10),    # Right arm
                (11, 13), (13, 15), # Left leg
                (12, 14), (14, 16), # Right leg
                (5, 6), (5, 11), (6, 12), (11, 12) # Torso
            ]
            
            # Draw skeleton
            for i, j in connections:
                if i < len(keypoints) and j < len(keypoints):
                    if confidences[i] > 0.5 and confidences[j] > 0.5:
                        pt1 = tuple(map(int, keypoints[i]))
                        pt2 = tuple(map(int, keypoints[j]))
                        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
            
            # Draw keypoints
            for i, (x, y) in enumerate(keypoints):
                if confidences[i] > 0.5:
                    cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            
            # Draw trajectory for right wrist (keypoint 10)
            if person['trajectory'] and len(person['trajectory']) > 1:
                wrist_idx = 10  # Right wrist in COCO format
                
                # Get trajectory points
                pts = []
                for hist in person['trajectory']:
                    if hist[wrist_idx][0] > 0 and hist[wrist_idx][1] > 0:
                        pts.append(tuple(map(int, hist[wrist_idx])))
                
                # Draw trajectory line
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        cv2.line(vis_frame, pts[i-1], pts[i], (255, 255, 0), 2)
        
        return vis_frame
    
    @staticmethod
    def setup_camera(camera_id: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
        """
        Set up and configure a camera for capturing.
        
        Args:
            camera_id: Camera device ID.
            width: Desired frame width.
            height: Desired frame height.
            
        Returns:
            Configured VideoCapture object.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {camera_id}")
            
        return cap