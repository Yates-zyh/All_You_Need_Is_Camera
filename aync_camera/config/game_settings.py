"""
游戏通用设置模块。
"""

# 屏幕设置
SCREEN_SETTINGS = {
    "default_width": 1280,
    "default_height": 720
}

# 游戏颜色
COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "green": (0, 200, 0),
    "red": (200, 0, 0),
    "blue": (0, 0, 200),
    "yellow": (255, 255, 0),
    "purple": (150, 0, 150),
    "cyan": (0, 200, 200),
    "gold": (255, 215, 0)  # 添加金色
}

# 关键点标签（用于显示）
KEYPOINT_LABELS = {
    0: "Nose",
    5: "L Shoulder",
    6: "R Shoulder",
    7: "L Elbow",
    8: "R Elbow",
    9: "L Wrist",
    10: "R Wrist",
    11: "L Hip",
    12: "R Hip",
    13: "L Knee",
    14: "R Knee",
    15: "L Ankle",
    16: "R Ankle",
}