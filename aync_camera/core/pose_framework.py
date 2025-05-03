"""
Core pose detection and tracking framework using YOLO11X-Pose.
"""
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PoseFramework:
    """YOLOv11x-Pose based framework for human pose detection and tracking."""

    def __init__(
        self,
        model_path: str = "yolov11x-pose.pt",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        track_buffer_size: int = 10,
    ):
        """
        Initialize the YOLOv11-Pose framework.

        Args:
            model_path: Path to the YOLOv11 pose model weights.
            device: Device to run the model on ('cuda' or 'cpu'). If None, will use cuda if available.
            confidence_threshold: Minimum detection confidence to consider.
            track_buffer_size: Number of past frames to track for trajectory analysis.
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.track_buffer_size = track_buffer_size
        
        # Tracking state
        self.tracked_poses = {}  # Dictionary to store state per tracked ID
        
        # Camera attribute (will be set when setup_camera is called)
        self.camera = None
        
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
        
        # Store the original frame for UI rendering
        original_frame = frame.copy()
        
        # Mirror the frame horizontally (flip left-right) for processing
        frame = cv2.flip(frame, 1)
        
        # Run YOLO11x-Pose inference
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
                    
                    # Swap left and right keypoints due to mirroring
                    # COCO keypoint format pairs to swap: (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
                    pairs_to_swap = [(5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
                    for left_idx, right_idx in pairs_to_swap:
                        if left_idx < len(person_data['keypoints_xy']) and right_idx < len(person_data['keypoints_xy']):
                            # Swap coordinates
                            person_data['keypoints_xy'][left_idx], person_data['keypoints_xy'][right_idx] = \
                                person_data['keypoints_xy'][right_idx], person_data['keypoints_xy'][left_idx]
                            # Swap confidences
                            person_data['keypoints_conf'][left_idx], person_data['keypoints_conf'][right_idx] = \
                                person_data['keypoints_conf'][right_idx], person_data['keypoints_conf'][left_idx]
                    
                    # Add to detected persons list if confidence is sufficient
                    if person_data['confidence'] > self.confidence_threshold:
                        detected_persons.append(person_data)
        
        updated_tracked_poses = {}
        
        # For each detected person, try to match with existing tracks
        for i, person in enumerate(detected_persons):
            track_id = 1 if i == 0 else i + 1
            
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
            'persons': detected_persons,
            'mirrored_frame': frame,       # The mirrored frame for pose display
            'original_frame': original_frame  # The original frame for UI rendering
        }
        
        return framework_output
    
    def draw_results(self, frame: np.ndarray, pose_data: Dict, highlight_palms: bool = True) -> np.ndarray:
        """
        Draw detection and tracking results on a frame.
        
        Args:
            frame: Input image frame.
            pose_data: Pose data from process_frame().
            highlight_palms: Whether to highlight palm positions with special markers
            
        Returns:
            Frame with visualizations.
        """
        # Use the mirrored frame if available
        if 'mirrored_frame' in pose_data:
            vis_frame = pose_data['mirrored_frame'].copy()
        else:
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
            
            # COCO keypoint connections (comprehensive skeleton)
            connections = [
                (5, 7), (7, 9),     # Left arm
                (6, 8), (8, 10),    # Right arm
                (11, 13), (13, 15), # Left leg
                (12, 14), (14, 16), # Right leg
                (5, 6),             # Shoulders
                (5, 11), (6, 12),   # Torso sides
                (11, 12),           # Hips
                (0, 1), (1, 2), (2, 3), (3, 4), # Face
                (0, 5), (0, 6)      # Neck
            ]
            
            # Draw skeleton with thicker lines
            for i, j in connections:
                if i < len(keypoints) and j < len(keypoints):
                    if confidences[i] > 0.5 and confidences[j] > 0.5:
                        pt1 = tuple(map(int, keypoints[i]))
                        pt2 = tuple(map(int, keypoints[j]))
                        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 3)
            
            # Draw all keypoints
            for i, (x, y) in enumerate(keypoints):
                if confidences[i] > 0.5:
                    # Regular keypoints
                    cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            
            # Highlight palm positions (wrists) with special markers
            if highlight_palms:
                # Left wrist (keypoint 9) and right wrist (keypoint 10)
                wrist_indices = [9, 10]
                for idx in wrist_indices:
                    if idx < len(keypoints) and confidences[idx] > 0.5:
                        x, y = map(int, keypoints[idx])
                        # Draw larger circle
                        cv2.circle(vis_frame, (x, y), 15, (0, 255, 255), 2)
                        # Draw cross inside
                        cv2.line(vis_frame, (x-10, y), (x+10, y), (0, 255, 255), 2)
                        cv2.line(vis_frame, (x, y-10), (x, y+10), (0, 255, 255), 2)
                        # Add label
                        label = "Left Palm" if idx == 9 else "Right Palm"
                        cv2.putText(
                            vis_frame,
                            label,
                            (x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
            
            # Draw trajectory for both wrists
            if person['trajectory'] and len(person['trajectory']) > 1:
                wrist_indices = [9, 10]  # Both left and right wrist
                colors = [(255, 255, 0), (0, 255, 255)]  # Yellow for left, cyan for right
                
                for wrist_idx, color in zip(wrist_indices, colors):
                    # Get trajectory points
                    pts = []
                    for hist in person['trajectory']:
                        if hist[wrist_idx][0] > 0 and hist[wrist_idx][1] > 0:
                            pts.append(tuple(map(int, hist[wrist_idx])))
                    
                    # Draw trajectory line
                    if len(pts) > 1:
                        for i in range(1, len(pts)):
                            cv2.line(vis_frame, pts[i-1], pts[i], color, 2)
        
        return vis_frame
    
    def setup_camera(self, camera_id: int = 0, width: int = 1920, height: int = 1080) -> None:
        """
        Set up and configure a camera for capturing.
        
        Args:
            camera_id: Camera device ID.
            width: Desired frame width.
            height: Desired frame height.
        """
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {camera_id}")
            
    def release(self) -> None:
        """
        Release the camera and other resources.
        This method should be called when the application ends.
        """
        if self.camera is not None:
            self.camera.release()
            self.camera = None