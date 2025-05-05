"""
Core pose detection and tracking framework using YOLO11X-Pose.
"""
import time
from collections import deque
from typing import Dict, Optional
import math
from math import radians, cos, sin

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
        user_height: float = 1.70,
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

        # User height (in meters)
        self.user_height = user_height  # Default height in meters
    
    def get_user_height(self):
        '''Prompt the user to enter their height.'''
        print("Please enter your height in cm:")
        while True:
            try:
                height = float(input())
                if height > 0:
                    return height
                else:
                    print("Height must be a positive number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    def calculate_distance(self, pixel_height, focal_length):
        '''use user height to calculate distance'''
        return (self.user_height * focal_length) / pixel_height
    
    def calculate_3d_angles(self, keypoints_np, focal_length):
        '''
        Calculate the 3D angles based on the pose landmarks.

        0: "nose", 5: "left_shoulder", 6: "right_shoulder", 
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 
        10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 
        16: "right_ankle"

        '''    
        angles = {'X': 0, 'Y': 0, 'Z': 0}

        # Extract keypoints
        nose = keypoints_np[0][0]
        left_shoulder = keypoints_np[0][5]
        right_shoulder = keypoints_np[0][6]
        left_hip = keypoints_np[0][11]
        right_hip = keypoints_np[0][12]
        left_ankle = keypoints_np[0][15]
        right_ankle = keypoints_np[0][16]

        # X-axis tilt (left/right)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        vertical_diff = shoulder_center - hip_center
        angles['X'] = math.degrees(math.atan(vertical_diff[0]/(vertical_diff[1]+0.00001)))

        # Y-axis tilt (forward/backward)
        shoulder_length = np.linalg.norm(right_shoulder - left_shoulder)
        shoulder_distance = (0.40 * focal_length) / shoulder_length#tune this value 0.4
        hip_length = np.linalg.norm(right_hip - left_hip)
        hip_distance = (0.30 * focal_length) / hip_length#tune this value 0.3
        height = nose[1]-left_ankle[1]  # Y-axis height difference
        angles['Y'] = math.degrees(math.cos((shoulder_distance**2 + hip_distance**2 - (self.user_height/2)**2)/(2*shoulder_distance*hip_distance)))  # Positive for forward lean

        # Z-axis rotation (shoulder vs. hip line)
        shoulder_vector = np.array([right_shoulder[0] - left_shoulder[0], 
                                    right_shoulder[1] - left_shoulder[1]])
        hip_vector = np.array([right_hip[0] - left_hip[0], 
                                right_hip[1] - left_hip[1]])
        cos_theta = np.dot(shoulder_vector, hip_vector) / (
            np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector) + 1e-7)
        angles['Z'] = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        return angles

    def get_rotation_matrix(pitch, yaw, roll):
        """Generate a 3D rotation matrix for pitch, yaw, and roll."""
        pitch = radians(pitch)
        yaw = radians(yaw)
        roll = radians(roll)

        # Rotation matrix for pitch (X-axis)
        Rx = np.array([[1, 0, 0],
                    [0, cos(pitch), -sin(pitch)],
                    [0, sin(pitch), cos(pitch)]])
        
        # Rotation matrix for yaw (Y-axis)
        Ry = np.array([[cos(yaw), 0, sin(yaw)],
                    [0, 1, 0],
                    [-sin(yaw), 0, cos(yaw)]])
        
        # Rotation matrix for roll (Z-axis)
        Rz = np.array([[cos(roll), -sin(roll), 0],
                    [sin(roll), cos(roll), 0],
                    [0, 0, 1]])

        # Combined rotation matrix
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
        
        return rotation_matrix

    def is_user_still(self, prev_positions, current_position, threshold=10):
        distance_moved = np.linalg.norm(np.array(prev_positions) - np.array(current_position))
        if distance_moved < threshold:
            return True
        return False

    def draw_instructions(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw instructions on the frame.
        
        Args:
            frame: Input image frame.
            instructions: Instructions to display.
            
        Returns:
            Frame with instructions drawn.
        """
        #print("Drawing instructions...")
        instruction_frame = frame.copy()
        # Draw instructions on the frame
        for person in pose_data['persons']:
            distance = person['distance']
            angles = person['angles']
            cv2.putText(instruction_frame, f"Distance: {distance:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(instruction_frame, f"X: {angles['X']:.2f} Y: {angles['Y']:.2f} Z: {angles['Z']:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if distance < 1.5:
                cv2.putText(instruction_frame, "Too Close! Move Back", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if distance > 3.0:
                cv2.putText(instruction_frame, "Too Far! Move Closer", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['X'] > 10:
                cv2.putText(instruction_frame, "Tilt Left", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['X'] < -10:
                cv2.putText(instruction_frame, "Tilt Right", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['Y'] > 10:
                cv2.putText(instruction_frame, "Lean Forward", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['Y'] < -10:
                cv2.putText(instruction_frame, "Lean Backward", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['Z'] > 10:
                cv2.putText(instruction_frame, "Rotate Left", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angles['Z'] < -10:
                cv2.putText(instruction_frame, "Rotate Right", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return instruction_frame
    
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

        focal_length = 1200  # Example focal length, adjust based on your camera setup
        
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
                        'trajectory': None,  # Will be filled by tracking
                        'distance': None,  # Will be filled by calculation_distance
                        'angles': [0,0,0]  # Placeholder for angles
                    }
                    # Calculate pixel height of the bounding box
                    y_min, x_min, y_max, x_max = boxes_np[i][:4]  # Extract the coordinates
                    pixel_height = y_max - y_min  # Calculate the pixel height
                    
                    # Calculate the distance to the person using the pixel height
                    distance = self.calculate_distance(pixel_height, focal_length)
                    person_data['distance'] = distance

                    # Calculate 3D angles based on the pose landmarks
                    person_data['angles'] = self.calculate_3d_angles(keypoints_np, focal_length)
                    
                    #print(f"Distance: {distance:.2f} m, Angles: {person_data['angles']}")

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
            # Define keypoint names for reference
            keypoint_names = {
                0: "nose", 5: "left_shoulder", 6: "right_shoulder", 
                7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 
                10: "right_wrist", 11: "left_hip", 12: "right_hip",
                13: "left_knee", 14: "right_knee", 15: "left_ankle", 
                16: "right_ankle"
            }
            
            # Define keypoint colors for different body parts
            keypoint_colors = {
                0: (255, 0, 0),      # nose - blue
                5: (0, 255, 255),    # left_shoulder - yellow
                6: (0, 255, 255),    # right_shoulder - yellow
                7: (0, 128, 255),    # left_elbow - orange
                8: (0, 128, 255),    # right_elbow - orange
                9: (0, 255, 255),    # left_wrist - yellow
                10: (0, 255, 255),   # right_wrist - yellow
                11: (255, 0, 255),   # left_hip - magenta
                12: (255, 0, 255),   # right_hip - magenta
                13: (255, 0, 128),   # left_knee - purple
                14: (255, 0, 128),   # right_knee - purple
                15: (128, 0, 255),   # left_ankle - violet
                16: (128, 0, 255)    # right_ankle - violet
            }
            
            # Get keypoints
            keypoints = person['keypoints_xy']
            confidences = person['keypoints_conf']
            
            # Draw each keypoint with its specific color and marker
            for i, (x, y) in enumerate(keypoints):
                if confidences[i] > 0.5:
                    # Get color for this keypoint
                    color = keypoint_colors.get(i, (0, 0, 255))  # default red
                    
                    # Draw keypoint with specific style based on body part type
                    if i == 0:  # nose
                        # Draw nose as diamond
                        points = np.array([
                            [x, y-6], [x+6, y], [x, y+6], [x-6, y]
                        ], np.int32)
                        cv2.polylines(vis_frame, [points], True, color, 2)
                    elif i in [5, 6]:  # shoulders
                        # Draw shoulders as squares
                        cv2.rectangle(vis_frame, (int(x)-5, int(y)-5), (int(x)+5, int(y)+5), color, 2)
                    elif i in [9, 10]:  # wrists
                        # Wrists are handled separately below, just draw a small point here
                        cv2.circle(vis_frame, (int(x), int(y)), 4, color, -1)
                    elif i in [15, 16]:  # ankles
                        # Draw ankles as triangles
                        points = np.array([
                            [x, y-7], [x+7, y+7], [x-7, y+7]
                        ], np.int32)
                        cv2.fillPoly(vis_frame, [points], color)
                    else:  # all other keypoints
                        # Draw as circles
                        cv2.circle(vis_frame, (int(x), int(y)), 4, color, -1)
                    
                    # Add keypoint label
                    if i not in [9, 10]:  # Don't add labels for wrists here (done separately)
                        label = keypoint_names.get(i, f"kp{i}")
                        # Only show shortened labels to avoid clutter
                        label_parts = label.split('_')
                        short_label = label_parts[0][0] + "_" + label_parts[1][0] if len(label_parts) > 1 else label[0]
                        cv2.putText(
                            vis_frame,
                            short_label,
                            (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1
                        )
            
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
                        # Add label in English
                        label = "Left Wrist" if idx == 9 else "Right Wrist"
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
    
    def setup_camera(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> None:
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