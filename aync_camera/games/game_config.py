"""
Game configuration module for the Falling-Note Rhythm Game.
"""

# Game difficulty presets
DIFFICULTY_PRESETS = {
    "easy": {
        "spawn_rate": 0.01,
        "note_speed": 3.0,
        "confidence_threshold": 0.4
    },
    "normal": {
        "spawn_rate": 0.02,
        "note_speed": 5.0,
        "confidence_threshold": 0.5
    },
    "hard": {
        "spawn_rate": 0.04,
        "note_speed": 7.0,
        "confidence_threshold": 0.6
    }
}

# Default lane to keypoint mapping (COCO format)
# Left Ankle = 15, Right Ankle = 16, Left Wrist = 9, Right Wrist = 10
DEFAULT_LANE_KEYPOINT_MAP = {
    0: 15,  # Left Ankle -> Lane 0
    1: 9,   # Left Wrist -> Lane 1
    2: 10,  # Right Wrist -> Lane 2
    3: 16   # Right Ankle -> Lane 3
}

# Keypoint labels for display
KEYPOINT_LABELS = {
    9: "L Hand",
    10: "R Hand",
    15: "L Foot",
    16: "R Foot"
}

# Game window settings
DEFAULT_SCREEN_WIDTH = 800
DEFAULT_SCREEN_HEIGHT = 600

# Game colors
COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "green": (0, 200, 0),
    "red": (200, 0, 0),
    "blue": (0, 0, 200),
    "yellow": (255, 255, 0),
    "purple": (150, 0, 150),
    "cyan": (0, 200, 200)
}

# Scoring system
SCORE_PER_HIT = 10
COMBO_MULTIPLIER = True  # If True, combo will multiply score