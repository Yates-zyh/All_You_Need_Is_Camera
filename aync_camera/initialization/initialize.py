'''
The initialization is responsible for setting up the camera and its parameters.
It includes functions to initialize the camera, set its resolution, and adjust its settings.
The module also includes functions to check if the camera is ready and to provide feedback to the user
during the initialization process.
should return the camera object and any other necessary parameters for further processing.
'''
import cv2
import numpy as np
import mediapipe as mp
from aync_camera.initialization.distance_pose_height import distance_detection, provide_distance_feedback
from aync_camera.initialization.angle import calculate_3d_angles, draw_angle_indicator

class Initialization:
    """
    A class to initialize the camera, detect distance, angles, and provide feedback to the user.
    """

    def __init__(self, camera_id=0, width=1280, height=720):
        """
        Initialize the Initialization class.
        
        Args:
            camera_id: The ID of the camera to use (default is 0).
            width: The width of the camera frame (default is 1280).
            height: The height of the camera frame (default is 720).
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.camera = None
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def open_camera(self):
        """Open the camera and set its resolution."""
        self.camera = cv2.VideoCapture(self.camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def get_user_height(self):
        """Prompt the user to enter their height."""
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
    
    def calculate_3d_angles(self, frame):
        """Calculate the 3D angles based on the pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        angles = {'X': 0, 'Y': 0, 'Z': 0}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract keypoints
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # X-axis tilt (left/right)
            shoulder_center = (left_shoulder.y + right_shoulder.y) / 2
            hip_center = (left_hip.y + right_hip.y) / 2
            vertical_diff = shoulder_center - hip_center
            angles['X'] = np.degrees(np.arcsin(vertical_diff * 2))  # Sensitivity scaling

            # Y-axis tilt (forward/backward)
            nose_projection = nose.y
            ankle_projection = left_ankle.y
            forward_lean = nose_projection - ankle_projection
            angles['Y'] = np.degrees(np.arcsin(forward_lean * 2))  # Positive for forward lean

            # Z-axis rotation (shoulder vs. hip line)
            shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, 
                                        right_shoulder.y - left_shoulder.y])
            hip_vector = np.array([right_hip.x - left_hip.x, 
                                   right_hip.y - left_hip.y])
            cos_theta = np.dot(shoulder_vector, hip_vector) / (
                np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector) + 1e-7)
            angles['Z'] = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        return angles
    
    def distance_detection(self, frame, user_height=170):
        """Detect distance from the camera based on pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        distance_text = "No Body Detected"
        distance = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Get head and toe landmarks
            head = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            head_px = (int(head.x * w), int(head.y * h))
            toes = ((left_ankle.x + right_ankle.x) / 2 * w, (left_ankle.y + right_ankle.y) / 2 * h)
            
            # Calculate pixel distance between head and toes
            pixel_distance = np.linalg.norm(np.array(head_px) - np.array(toes))
            
            if pixel_distance == 0:
                distance_text = "Error: Zero pixel distance detected"
            else:
                focal_length = 1200  # Example focal length
                distance = (user_height * focal_length) / pixel_distance
                distance_text = f"Distance: {distance:.2f} meters"

            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Overlay distance text
        cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, distance, distance_text

    def main(self):
        """Main loop to process video stream and provide feedback."""
        self.open_camera()
        user_height = self.get_user_height()
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Get distance and angles
            
            frame, distance, distance_text = self.distance_detection(frame, user_height)
            angles = self.calculate_3d_angles(frame)
            
            # Show the distance and angles on the screen
            cv2.putText(frame, f"Distance: {distance_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Angles: X={angles['X']:.2f} Y={angles['Y']:.2f} Z={angles['Z']:.2f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display the result
            cv2.imshow("Initialization - Adjust Position", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    init = Initialization()
    init.main()