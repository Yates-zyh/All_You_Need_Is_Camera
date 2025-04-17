from scipy.spatial.distance import cosine
import numpy as np

def normalize_pose(keypoints):
    """Normalizes pose keypoints relative to a central point (e.g., pelvis) and scale."""
    # Example: Use midpoint between hips as center (indices depend on model)
    # Assume left_hip=11, right_hip=12
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    center = (left_hip + right_hip) / 2.0

    # Translate keypoints so center is at origin
    normalized_kps = keypoints - center

    # Normalize scale (e.g., based on torso length: shoulder-hip distance)
    # Assume left_shoulder=5, right_shoulder=6
    left_shoulder = normalized_kps[5]
    right_shoulder = normalized_kps[6]
    # Use distance between shoulder midpoint and hip midpoint as scale reference
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    torso_length = np.linalg.norm(shoulder_mid) # Distance from origin (hip mid)
    if torso_length > 1e-6: # Avoid division by zero
        normalized_kps /= torso_length
    else: # Handle case where torso length is zero (unlikely)
        # Find max distance from center and normalize by that? Or return zeros?
        max_dist = np.max(np.linalg.norm(normalized_kps, axis=1))
        if max_dist > 1e-6:
            normalized_kps /= max_dist

    return normalized_kps.flatten() # Return as a flat vector for similarity calculation

def calculate_pose_similarity(user_kps_xy, template_kps_xy):
    """Calculates similarity between two normalized pose vectors."""
    if len(user_kps_xy) == 0 or len(template_kps_xy) == 0:
        return 0.0

    # Ensure only valid keypoints are used (e.g., filter based on confidence before normalization)
    # Assume keypoints are already filtered and normalized

    # Normalize both poses
    user_norm_flat = normalize_pose(user_kps_xy)
    template_norm_flat = normalize_pose(template_kps_xy)

    # Calculate Cosine Similarity
    # Ensure vectors have the same dimension
    if len(user_norm_flat) != len(template_norm_flat):
        print("Warning: Pose dimensions mismatch!")
        return 0.0

    # Cosine distance is 1 - similarity
    similarity = 1.0 - cosine(user_norm_flat, template_norm_flat)

    # Clamp similarity to [0, 1]
    return max(0.0, min(1.0, similarity))

# --- In Game Loop ---
# Get user_keypoints = pose_data['persons'][0]['keypoints_xy']
# Load current template_pose_keypoints based on song time/beat
# Filter keypoints based on confidence first
# confident_user_kps = user_keypoints[user_conf > 0.5]
# confident_template_kps = template_pose_keypoints[template_conf > 0.5] # Assuming template has conf too
# similarity_score = calculate_pose_similarity(confident_user_kps, confident_template_kps)
# Display score / visual feedback