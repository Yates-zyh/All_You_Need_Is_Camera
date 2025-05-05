"""
节奏游戏逻辑模块，处理游戏核心玩法逻辑。
"""
import random
import time
import numpy as np

from aync_camera.config.rhythm_config import (COMBO_MULTIPLIER, 
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
        self.note_radius = 30  # 内圈半径 (减小为30，原值为40)
        self.max_radius = 100  # 初始判定圈半径 (减小为100，原值为120)
        self.acceptance_radius = 60  # 手部距离音符中心多近算命中 (调整为60，原值为70)
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
        
        # 坐标映射参数
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.prev_position = None  # To store previous position of the user
        self.is_still = False
        self.is_moving = True  # To track if the user is moving
        self.still_start_time = None  # Time when the user started staying still
        self.still_duration = 3  # Duration to stay still in seconds (3 seconds)

        # Initialize the music sheet loader
        self.is_ready = False
        
        # 如果提供了谱面路径，尝试加载
        if self.music_sheet_path:
            self.using_music_sheet = self.music_sheet_loader.load_music_sheet(self.music_sheet_path)
            # 如果加载成功，计算映射参数
            if self.using_music_sheet and 'video_info' in self.music_sheet_loader.json_data:
                self._calculate_mapping_params()
        
        # 游戏开始时间
        self.game_start_time = time.time()
        
    def _calculate_mapping_params(self):
        """计算JSON坐标到屏幕坐标的映射参数"""
        # 获取原始视频尺寸
        video_info = self.music_sheet_loader.json_data.get('video_info', {})
        json_video_width = video_info.get('width', self.screen_width)
        json_video_height = video_info.get('height', self.screen_height)
        
        # 计算缩放因子
        scale_x = self.screen_width / json_video_width
        scale_y = self.screen_height / json_video_height
        self.scale_factor = min(scale_x, scale_y)
        
        # 计算缩放后的尺寸
        scaled_width = json_video_width * self.scale_factor
        scaled_height = json_video_height * self.scale_factor
        
        # 计算居中偏移量
        self.offset_x = (self.screen_width - scaled_width) / 2
        self.offset_y = (self.screen_height - scaled_height) / 2
        
    def _map_coordinates(self, x, y):
        """
        将JSON坐标映射到屏幕坐标
        
        Args:
            x: JSON中的X坐标
            y: JSON中的Y坐标
            
        Returns:
            tuple: (screen_x, screen_y)
        """
        # 先应用缩放
        screen_x = x * self.scale_factor
        screen_y = y * self.scale_factor
        
        # 再应用偏移
        screen_x += self.offset_x
        screen_y += self.offset_y
        
        return int(screen_x), int(screen_y)
    
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
                
                # 映射坐标到屏幕范围内
                x, y = self._map_coordinates(x, y)
                
                # 调整坐标到屏幕范围内
                x = min(max(x, self.padding), self.screen_width - self.padding)
                y = min(max(y, self.padding), self.screen_height - self.padding)
                
                # 获取该关键点对应的身体部位名称和索引
                keypoint_name = None
                keypoint_index = None
                
                if idx < len(self.music_sheet_loader.keypoint_names):
                    keypoint_name = self.music_sheet_loader.keypoint_names[idx]
                
                if idx < len(self.music_sheet_loader.keypoint_indices):
                    keypoint_index = self.music_sheet_loader.keypoint_indices[idx]
                else:
                    # 如果没有明确的索引映射，使用与名称对应的索引
                    # 这里假设索引与JSON中的顺序对应
                    keypoint_index = idx
                
                # 创建新音符
                note = Note(
                    x=int(x),
                    y=int(y),
                    radius=self.note_radius,
                    max_radius=self.max_radius,
                    duration=self.note_duration,
                    keypoint_name=keypoint_name,
                    keypoint_index=keypoint_index
                )
                self.notes.append(note)
            
            # 处理完这个节拍后，移动到下一个
            self.current_beat_index += 1
    
    def check_ready(self, pose_data):
        """
        Check if the player's hand is over the 'Ready' button.

        Args:
            pose_data (dict): The pose data containing keypoints for the user.
            
        Returns:
            bool: True if the hand is over the button, otherwise False.
        """
        #print("Checking if hand is over the button...")
        # Define the button position and size on the screen (example)
        button_x, button_y = self.screen_width // 2, self.screen_height // 2  # Center of the screen
        button_width, button_height = 300, 60  # Size of the button (width, height)
        
        # Define the button's area (top-left and bottom-right corners)
        button_left = button_x - button_width // 2
        button_top = button_y - button_height // 2
        button_right = button_x + button_width // 2
        button_bottom = button_y + button_height // 2

        # Check if any hand (left or right wrist) is within the button area
        if pose_data['persons']:
            person = pose_data['persons'][0]
            keypoints = person['keypoints_xy']
            keypoints_conf = person['keypoints_conf']

            # Get the left and right wrist positions (index 9 and 10 in COCO keypoints)
            if len(keypoints) > 10:
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]

                # Check if left wrist is over the button
                if self.is_within_button_area(left_wrist[0], left_wrist[1], button_left, button_top, button_right, button_bottom):
                    #print("Left wrist is over the button")
                    self.is_ready = True
                    return True
                # Check if right wrist is over the button
                if self.is_within_button_area(right_wrist[0], right_wrist[1], button_left, button_top, button_right, button_bottom):
                    #print("right wrist is over the button")
                    self.is_ready = True
                    return True
        return False

    def is_within_button_area(self, x, y, button_left, button_top, button_right, button_bottom):
        """
        Check if a point (x, y) is inside the button area.
        
        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
            button_left (int): Left side of the button.
            button_top (int): Top side of the button.
            button_right (int): Right side of the button.
            button_bottom (int): Bottom side of the button.
            
        Returns:
            bool: True if the point is within the button area, otherwise False.
        """
        return button_left <= x <= button_right and button_top <= y <= button_bottom

    def is_user_still(self, current_position, threshold=10):
        if self.prev_position is None:
            return False  # If there's no previous position, we assume movement

        distance_moved = np.linalg.norm(np.array(self.prev_position) - np.array(current_position))
        #print("Distance moved:", distance_moved)
        if distance_moved < threshold:
            return True
        return False
    
    def check_still(self, pose_data, threshold=10):
        """
        Check if the user is still (not moving significantly).
        
        Args:
            pose_data: Data from Pose framework that contains the keypoints.
            threshold: The threshold to determine if the user is still.
            
        Returns:
            bool: True if the user is still, False otherwise.
        """
        if not pose_data['persons']:
            return False
        person = pose_data['persons'][0]
        keypoints = person['keypoints_xy']

        head = keypoints[0]

        # Check if the user is still (not moving significantly)
        if self.prev_position is not None:
            #print("Previous position:", self.prev_position)
            #print("Current head position:", head)
            still = self.is_user_still(head, threshold)
            if still:
                self.is_moving = False
                if self.still_start_time is None:
                    # User just became still, start tracking time
                    self.still_start_time = time.time()
                else:
                    elapsed_time = time.time() - self.still_start_time
                    if elapsed_time >= self.still_duration:
                        self.is_still = True
            else:
                print("User is moving, please stay still.")
                self.still_start_time = None  # Reset the timer
                self.is_still = False
                self.is_moving = True
        # Update the previous position for the next frame
        self.prev_position = head

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
        
        # 定义所有可用于交互的关键点索引
        # 0: nose, 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow, 
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        active_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        # 定义关键点名称以便参考
        keypoint_names = {
            0: "nose", 5: "left_shoulder", 6: "right_shoulder", 
            7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 
            10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 
            16: "right_ankle"
        }
        
        # 收集所有有效的关键点位置
        keypoint_positions = {}
        for idx in active_keypoints:
            if idx < len(keypoints) and keypoints_conf[idx] >= self.confidence_threshold:
                keypoint_positions[idx] = (int(keypoints[idx][0]), int(keypoints[idx][1]))
        
        # 检查每个音符
        for note in self.notes:
            if note.judged:
                continue
            
            # 检查音符是否与任何关键点发生碰撞
            hit = False
            hit_kp_idx = None  # 记录击中的关键点索引
            hit_distance = float('inf')  # 记录最近的击中距离

            # 首先检查当前音符需要哪个关键点
            required_kp_idx = note.keypoint_index
            required_kp_name = note.keypoint_name
            
            # 如果需要特定关键点索引，并且是成对的左右关键点之一，执行镜像映射
            # COCO关键点对应关系：左右肩(5,6), 左右肘(7,8), 左右腕(9,10), 左右髋(11,12), 左右膝(13,14), 左右踝(15,16)
            pairs_mapping = {5:6, 6:5, 7:8, 8:7, 9:10, 10:9, 11:12, 12:11, 13:14, 14:13, 15:16, 16:15}
            
            # 如果是身体左右对称的关键点，进行镜像映射
            if required_kp_idx in pairs_mapping:
                required_kp_idx = pairs_mapping[required_kp_idx]
            
            # 同样对关键点名称进行镜像映射
            if required_kp_name:
                name_parts = required_kp_name.split('_')
                if len(name_parts) > 1 and name_parts[0] in ["left", "right"]:
                    # 交换左右
                    if name_parts[0] == "left":
                        name_parts[0] = "right"
                    else:
                        name_parts[0] = "left"
                    required_kp_name = "_".join(name_parts)
            
            for kp_idx, (px, py) in keypoint_positions.items():
                # 计算关键点到音符中心的距离
                distance = ((note.x - px) ** 2 + (note.y - py) ** 2) ** 0.5
                
                # 检查关键点是否在音符接受半径内
                if distance <= self.acceptance_radius:
                    # 如果音符需要特定关键点
                    if required_kp_idx is not None:
                        if kp_idx == required_kp_idx and distance < hit_distance:
                            hit = True
                            hit_kp_idx = kp_idx
                            hit_distance = distance
                    elif required_kp_name is not None:
                        # 检查关键点名称是否匹配
                        keypoint_name = keypoint_names.get(kp_idx)
                        if keypoint_name == required_kp_name and distance < hit_distance:
                            hit = True
                            hit_kp_idx = kp_idx
                            hit_distance = distance
                    else:
                        # 如果音符没有特定要求，任何关键点都可以触发
                        if distance < hit_distance:
                            hit = True
                            hit_kp_idx = kp_idx
                            hit_distance = distance
            
            # 处理命中结果
            if hit and hit_kp_idx is not None:
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
            
            # 检查音符是否被错过（判定时机到达但未被命中）
            elif note.update():
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