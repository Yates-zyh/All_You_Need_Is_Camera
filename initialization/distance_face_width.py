import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Known real-world average face width (in meters)
REAL_FACE_WIDTH = 0.15

# Focal length calculation (example values - should be calibrated for your camera)
# Focal length (in pixels) can be calculated using: (face_width_pixels * known_distance) / real_face_width
# Example calibration: If face width is 100px at 1.5m, focal_length_px = (100 * 1.5) / 0.15 = 1000
FOCAL_LENGTH_PX = 1000  # Update this based on your camera calibration

def calculate_distance(face_width_px):
    """Calculate real-world distance from camera using face width."""
    if face_width_px is None or face_width_px == 0:
        return None
    return (REAL_FACE_WIDTH * FOCAL_LENGTH_PX) / face_width_px

def main():
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            # Get first face detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            # Calculate face width in pixels
            face_width_px = bbox.width * iw
            distance = calculate_distance(face_width_px)
            
            if distance is not None:
                # Determine user feedback
                if distance < 1.5:
                    status = f"Too Close: {distance:.2f}m - Move Back"
                elif distance > 3.0:
                    status = f"Too Far: {distance:.2f}m - Move Closer"
                else:
                    status = f"Good Distance: {distance:.2f}m"
                
                # Draw face bounding box
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            status = "No face detected"
        
        # Display status
        cv2.putText(frame, status, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Distance Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()