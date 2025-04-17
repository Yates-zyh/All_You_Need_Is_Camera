import pygame
from collections import deque

# --- In Game Loop ---
# Get pose_data = framework.process_frame(...)
# if pose_data['persons']:
#     person = pose_data['persons'][0]
#     if person['trajectory'] and len(person['trajectory']) >= 2:
#         # Get trajectory for right hand (e.g., wrist index 10)
#         hand_idx = 10
#         # Ensure keypoint confidence is high enough in the trajectory points
#         # Extract last two points of the trajectory for the hand
#         pt1 = person['trajectory'][-2][hand_idx]
#         pt2 = person['trajectory'][-1][hand_idx]
#
#         # Check if points are valid (e.g., confidence was high)
#         # ... (Confidence check omitted for brevity) ...
#
#         # Represent trajectory as a line segment (pt1, pt2)
#
#         # Iterate through active fruit/bomb sprites
#         for fruit in active_fruits[:]:
#             # Check collision between line segment (pt1, pt2) and fruit.rect
#             # Pygame has limited line collision; might need line-rect intersection algorithm
#             if line_segment_intersects_rect(pt1, pt2, fruit.rect):
#                  print("Slice!")
#                  # Remove fruit, add score, play sound/effect
#                  active_fruits.remove(fruit)
#                  score += 1
#
# def line_segment_intersects_rect(p1, p2, rect):
#      """Placeholder for line segment-rectangle intersection test."""
#      # Implementation using Pygame's rect.clipline or custom geometry checks needed
#      # rect.clipline might work: rect.clipline(p1, p2) returns points inside rect
#      clipped_line = rect.clipline(p1, p2)
#      return len(clipped_line) > 0 # Returns True if any part of the line is inside