'''
The initialization module is responsible for setting up the camera and its parameters.
It includes functions to initialize the camera, set its resolution, and adjust its settings.
The module also includes functions to check if the camera is ready and to provide feedback to the user
during the initialization process.
should return the camera object and any other necessary parameters for further processing.
'''
import cv2
import numpy as np
import mediapipe as mp
from distance_pose_height import distance_detection, provide_distance_feedback
from angle import calculate_3d_angles, draw_angle_indicator

def get_user_height():
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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    user_height = get_user_height()/100  # Convert height to meters
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks:
            processed_frame, distance, distance_text = distance_detection(frame_rgb, pose, mp_pose, mp_drawing, user_height)
            angles = calculate_3d_angles(frame,pose, mp_pose, mp_drawing)


            # Provide real-time feedback based on the distance
            feedback = provide_distance_feedback(distance)
            cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    # Get user height
    
    # Example usage
    video_path = '/Users/macbookair/Downloads/IMG_0505.MOV'
    process_video(video_path)
