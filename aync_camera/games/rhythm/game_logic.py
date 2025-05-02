"""
节奏游戏逻辑模块，处理游戏核心玩法逻辑。
"""
import random
import time
from typing import Dict, List, Optional, Tuple, Union

from aync_camera.config.rhythm_config import (COMBO_MULTIPLIER, DEFAULT_LANE_KEYPOINT_MAP,
                                             DIFFICULTY_PRESETS, SCORE_PER_HIT)
from aync_camera.games.rhythm.note import Note
from aync_camera.games.rhythm.music_sheet_loader import MusicSheetLoader


class RhythmGameLogic:
    """处理节奏游戏的核心玩法逻辑。"""
    
    def __init__(
        self, 
        difficulty: str = "normal",
        screen_width: int = 1280,
        screen_height: int = 720,
        music: str = None,
        music_sheet_path: str = None
    ):
        """
        初始化游戏逻辑。
        
        Args:
            difficulty: 游戏难度（'easy', 'normal', 'hard'）
            screen_width: 游戏窗口宽度，用于计算音符生成范围
            screen_height: 游戏窗口高度，用于计算音符生成范围
            music: 要使用的音乐名称（例如 'earthquake', 'maria'）
            music_sheet_path: 包含节拍数据的JSON文件路径（可选）
                             如果提供，会覆盖music和difficulty设置
        """
        # 加载难度设置
        self.difficulty = difficulty
        self.difficulty_settings = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])
        
        # 游戏区域设置
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.note_radius = 40  # 内圈半径
        self.max_radius = 120  # 初始判定圈半径
        self.acceptance_radius = 70  # 手部距离音符中心多近算命中
        self.padding = 100  # 边缘填充，防止音符生成在屏幕边缘
        
        # 计算音符持续时间（圈的收缩速度）
        self.note_duration = max(3.0, 8.0 - self.difficulty_settings["note_speed"])
        
        # 设置置信度阈值，用于姿态检测
        self.confidence_threshold = self.difficulty_settings["confidence_threshold"]
        
        # 游戏状态
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        
        # 活跃的音符
        self.notes = []
        
        # 命中反馈（命中成功的视觉指示器）
        self.hit_feedback = []  # (x, y, time, hit_type) 元组列表，用于命中动画
        self.hit_feedback_duration = 0.5  # 显示命中反馈的持续时间（秒）
        
        # 音乐谱面
        self.music = music
        self.music_sheet_path = music_sheet_path
        self.music_sheet_loader = MusicSheetLoader()
        self.using_music_sheet = False
        self.current_beat_index = 0
        
        # 如果提供了谱面路径，尝试加载
        if self.music_sheet_path:
            self.using_music_sheet = self.music_sheet_loader.load_music_sheet(self.music_sheet_path)
        
        # 游戏开始时间
        self.game_start_time = time.time()
        
    def reset(self):
        """重置游戏状态。"""
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.notes = []
        self.hit_feedback = []
        self.current_beat_index = 0
        self.game_start_time = time.time()
        
    def spawn_note(self):
        """在屏幕上随机位置生成一个新音符。"""
        # 在屏幕范围内随机生成位置（考虑边缘填充）
        x = random.randint(self.padding, self.screen_width - self.padding)
        y = random.randint(self.padding, self.screen_height - self.padding)
        
        note = Note(
            x=x,
            y=y,
            radius=self.note_radius,
            max_radius=self.max_radius,
            duration=self.note_duration
        )
        self.notes.append(note)
    
    def spawn_notes_from_music_sheet(self, elapsed_time):
        """
        根据音乐谱面数据和当前经过的时间生成音符。
        
        Args:
            elapsed_time: 从游戏开始到现在经过的时间（秒）
        """
        # 如果没有加载音乐谱面或已处理完所有节拍，则返回
        if not self.using_music_sheet or self.current_beat_index >= len(self.music_sheet_loader.beat_data):
            return
        
        # 获取当前需要处理的节拍
        while self.current_beat_index < len(self.music_sheet_loader.beat_data):
            current_beat = self.music_sheet_loader.beat_data[self.current_beat_index]
            beat_time = current_beat.get('time', 0)
            
            # 如果当前时间还没到达这个节拍时间，跳出循环
            if elapsed_time < beat_time:
                break
                
            # 如果当前时间已经超过了这个节拍时间，生成音符
            keypoints_data = current_beat.get('keypoints', [])
            
            # 遍历每个关键点坐标，如果有有效坐标则生成音符
            for idx, keypoint in enumerate(keypoints_data):
                # 检查是否有有效坐标
                if not keypoint or len(keypoint) == 0:
                    continue
                    
                # 确保坐标有效（不是[0, 0]或空值）
                if not keypoint[0] or (len(keypoint) > 1 and not keypoint[1]):
                    continue
                    
                # 获取关键点坐标
                if len(keypoint) == 1:  # 有些JSON格式是[x]而不是[x, y]
                    x = keypoint[0]
                    y = 400  # 默认高度
                else:
                    x = keypoint[0]
                    y = keypoint[1]
                
                # 调整坐标到屏幕范围内
                x = min(max(x, self.padding), self.screen_width - self.padding)
                y = min(max(y, self.padding), self.screen_height - self.padding)
                
                # 获取该关键点对应的身体部位名称
                keypoint_name = None
                if idx < len(self.music_sheet_loader.keypoint_names):
                    keypoint_name = self.music_sheet_loader.keypoint_names[idx]
                
                # 创建新音符
                note = Note(
                    x=int(x),
                    y=int(y),
                    radius=self.note_radius,
                    max_radius=self.max_radius,
                    duration=self.note_duration,
                    keypoint_name=keypoint_name
                )
                self.notes.append(note)
            
            # 处理完这个节拍后，移动到下一个
            self.current_beat_index += 1
    
    def check_hits(self, pose_data):
        """
        检查玩家姿态与音符之间的碰撞。
        
        Args:
            pose_data: 来自PoseFramework的姿态数据
        """
        if not pose_data['persons']:
            return
            
        # 获取第一个检测到的人
        person = pose_data['persons'][0]
        keypoints = person['keypoints_xy']
        keypoints_conf = person['keypoints_conf']
        
        # 只考虑手腕关键点（9: 左手腕, 10: 右手腕）
        wrist_indices = [9, 10]
        palm_positions = {}
        
        # 获取手掌位置，考虑置信度阈值
        for idx in wrist_indices:
            if idx < len(keypoints) and keypoints_conf[idx] >= self.confidence_threshold:
                palm_positions[idx] = (int(keypoints[idx][0]), int(keypoints[idx][1]))
        
        # 检查每个音符
        for note in self.notes:
            if note.judged:
                continue
            
            # 检查音符与手掌的碰撞
            for idx in wrist_indices:
                if idx not in palm_positions:
                    continue
                
                # 获取手掌位置
                px, py = palm_positions[idx]
                
                # 计算手掌到音符中心的距离
                distance = ((note.x - px) ** 2 + (note.y - py) ** 2) ** 0.5
                
                # 检查手掌是否在音符接受半径内
                if distance <= self.acceptance_radius:
                    # 命中！
                    self.hits += 1
                    
                    # 计算得分（如果启用连击倍率）
                    hit_score = SCORE_PER_HIT
                    if COMBO_MULTIPLIER:
                        hit_score *= max(1, self.combo)
                    self.score += hit_score
                    
                    # 更新连击
                    self.combo += 1
                    self.max_combo = max(self.max_combo, self.combo)
                    
                    # 标记音符为命中
                    note.hit = True
                    note.judged = True  # 标记这个音符已被判定
                    
                    # 添加视觉命中反馈
                    note.hit_feedback_time = time.time()
                    self.hit_feedback.append((note.x, note.y, note.hit_feedback_time, "hit"))
                    break
                
            # 检查音符是否被错过（判定时机到达但未被命中）
            if not note.hit and note.update():
                # 音符被错过
                self.misses += 1
                self.combo = 0
                note.missed = True
                note.judged = True  # 标记这个音符已被判定

    def update_notes(self):
        """更新所有音符状态，移除需要移除的音符。"""
        # 创建要移除的音符列表
        notes_to_remove = []
        
        for note in self.notes:
            # 如果音符已经完全淡出或判定完成，可能需要移除
            if note.update():
                notes_to_remove.append(note)
        
        # 移除需要移除的音符
        for note in notes_to_remove:
            if note in self.notes:
                self.notes.remove(note)
    
    def update_hit_feedback(self):
        """更新命中反馈动画。"""
        current_time = time.time()
        
        # 移除过期的命中反馈
        self.hit_feedback = [(x, y, time, hit_type) for x, y, time, hit_type in self.hit_feedback 
                            if current_time - time <= self.hit_feedback_duration]
    
    def update(self, elapsed_time=None):
        """
        更新游戏状态。
        
        Args:
            elapsed_time: 可选，从游戏开始到现在的时间（秒）。如果不提供，将使用内部计时。
        """
        if elapsed_time is None:
            elapsed_time = time.time() - self.game_start_time
        
        # 从音乐谱面生成音符
        if self.using_music_sheet:
            self.spawn_notes_from_music_sheet(elapsed_time)
        
        # 更新命中反馈
        self.update_hit_feedback()
    
    def get_game_state(self):
        """
        获取当前游戏状态，用于UI渲染。
        
        Returns:
            dict: 包含游戏状态信息的字典
        """
        return {
            'score': self.score,
            'combo': self.combo,
            'max_combo': self.max_combo,
            'hits': self.hits,
            'misses': self.misses,
            'notes': self.notes,
            'hit_feedback': self.hit_feedback,
            'hit_feedback_duration': self.hit_feedback_duration,
            'difficulty': self.difficulty
        }