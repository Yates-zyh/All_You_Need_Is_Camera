"""
Simple test script to verify the PoseFramework functionality.
Run this script to test if the framework is correctly installed and configured.
"""
import time
import cv2

from aync_camera.core.pose_framework import PoseFramework


def test_framework():
    """Test the basic functionality of the PoseFramework."""
    print("Testing PoseFramework...")
    
    # Initialize the framework
    try:
        framework = PoseFramework(
            model_path="yolov8n-pose.pt",  # Using the smallest model for quick testing
            confidence_threshold=0.5
        )
        print("✓ Framework initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize framework: {str(e)}")
        return False
    
    # Setup camera
    try:
        cap = framework.setup_camera(camera_id=0, width=640, height=480)
        print("✓ Camera initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize camera: {str(e)}")
        return False
    
    # Process a few frames to test
    try:
        print("Processing frames (press 'q' to exit)...")
        fps_history = []
        start_time = time.time()
        frame_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to capture frame")
                break
            
            # Process frame
            process_start = time.time()
            pose_data = framework.process_frame(frame)
            process_time = time.time() - process_start
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed
                fps_history.append(fps)
                print(f"FPS: {fps:.1f}, Processing time: {process_time*1000:.1f}ms")
                frame_count = 0
                start_time = time.time()
            
            # Draw results
            vis_frame = framework.draw_results(frame, pose_data)
            
            # Display statistics
            stats_text = f"FPS: {fps_history[-1]:.1f}" if fps_history else "FPS: --"
            cv2.putText(
                vis_frame,
                stats_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display person count
            persons_text = f"Persons: {len(pose_data['persons'])}"
            cv2.putText(
                vis_frame,
                persons_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show the frame
            cv2.imshow("PoseFramework Test", vis_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print results
        if fps_history:
            avg_fps = sum(fps_history) / len(fps_history)
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("✓ Frame processing test completed")
    except Exception as e:
        print(f"✗ Error during frame processing: {str(e)}")
        return False
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    success = test_framework()
    if success:
        print("All tests passed! The framework is working correctly.")
    else:
        print("Some tests failed. Please check the errors above.")