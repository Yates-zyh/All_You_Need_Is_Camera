"""
节奏游戏配置模块。
"""

# 游戏难度预设
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

# 默认音乐谱面路径
MUSIC_SHEETS = {
    "earthquake": {
        "easy": "musicsheet/earthquake/earthquake_easy.json",
        "normal": "musicsheet/earthquake/earthquake_normal.json",
        "hard": "musicsheet/earthquake/earthquake_hard.json"
    },
    "maria": {
        "easy": "musicsheet/maria/maria_easy.json",
        "normal": "musicsheet/maria/maria_normal.json",
        "hard": "musicsheet/maria/maria_hard.json"
    }
}

# 默认使用的音乐
DEFAULT_MUSIC = "earthquake"

# 计分系统
SCORE_PER_HIT = 10
COMBO_MULTIPLIER = True  # 如果为True，连击会增加得分倍数