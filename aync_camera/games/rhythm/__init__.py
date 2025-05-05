import time
import pygame
import logging
import os
import subprocess
import tempfile
import cv2

from aync_camera.common.game_base import GameBase
from aync_camera.games.rhythm.game_logic import RhythmGameLogic
from aync_camera.ui.rhythm.renderer import (RhythmGameRenderer, START_SCREEN, 
                                           SONG_SELECTION, COUNTDOWN, PLAYING, 
                                           PAUSED, GAME_OVER, INSTRUCTION_INIT, INITIALIZATION)
from aync_camera.config.rhythm_config import MUSIC_SHEETS, DEFAULT_MUSIC
#from aync_camera.initialization.initialize import Initialization

# 初始化pygame音频
pygame.mixer.init()

# Configure simple logging to track state transitions
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RhythmGame")


class RhythmGame(GameBase):
    """节奏游戏，使用身体姿态作为控制。"""
    
    def __init__(
        self, 
        camera_id: int = 0, 
        model_path: str = "yolo11x-pose.pt",
        difficulty: str = "normal",
        music: str = None,
        screen_width: int = 1280,
        screen_height: int = 720,
        music_sheet_path: str = None
    ):
        """
        初始化游戏。
        
        Args:
            camera_id: 摄像头设备ID
            model_path: YOLO11x-Pose模型路径
            difficulty: 游戏难度（'easy', 'normal', 'hard'）
            music: 要使用的音乐名称（例如 'earthquake', 'maria'）
            screen_width: 游戏窗口宽度
            screen_height: 游戏窗口高度
            music_sheet_path: 包含节拍数据的JSON文件路径（可选）
                             如果提供，会覆盖music和difficulty设置
        """
        # 调用基类初始化
        super().__init__(
            camera_id=camera_id,
            model_path=model_path,
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        # 设置窗口标题
        pygame.display.set_caption(f"Rhythm Game")
        
        # 存储游戏参数
        self.difficulty = difficulty
        self.music = music if music else DEFAULT_MUSIC
        self.music_sheet_path = music_sheet_path
        
        # 确定音乐谱面路径
        if not music_sheet_path and self.music in MUSIC_SHEETS:
            self.music_sheet_path = MUSIC_SHEETS[self.music][self.difficulty]
        
        # 游戏状态控制
        self.game_state = START_SCREEN
        self.countdown_start_time = None
        self.game_start_time = None
        
        # 音频相关属性
        self.audio_path = None
        self.audio_extracted = False
        self.audio_playing = False
        
        # 游戏结束标志
        self.game_finished = False
        self.remaining_notes = 0
        self.last_note_time = 0
        
        # 创建UI渲染器实例
        self.ui_renderer = RhythmGameRenderer(
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        # 预先创建游戏逻辑实例，避免不初始化导致的问题
        self.create_game_logic()
        
        logger.info(f"Game initialized with music: {self.music}, difficulty: {self.difficulty}")
        logger.info(f"Music sheet path: {self.music_sheet_path}")
    
    def initialize(self):
        self.initialization = Initialization(
            camera_id=self.camera_id,
            width=self.screen_width,
            height=self.screen_height
        )
        self.initialization.open_camera()
        user_height = self.initialization.get_user_height()

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Get distance and angles
            
            frame, distance, distance_text = self.distance_detection(frame, user_height)
            angles = self.calculate_3d_angles(frame)
            
            # Show the distance and angles on the screen
            cv2.putText(frame, f"Distance: {distance_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Angles: X={angles['X']:.2f} Y={angles['Y']:.2f} Z={angles['Z']:.2f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display the result
            cv2.imshow("Initialization - Adjust Position", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
    def create_game_logic(self):
        """创建游戏逻辑实例，在选择难度后调用"""
        self.game_logic = RhythmGameLogic(
            difficulty=self.difficulty,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            music=self.music,
            music_sheet_path=self.music_sheet_path
        )
        logger.info(f"Game logic created with difficulty: {self.difficulty}")
        if self.game_logic.using_music_sheet:
            logger.info(f"Music sheet loaded successfully: {self.game_logic.music_sheet_path}")
            logger.info(f"Number of beats: {len(self.game_logic.music_sheet_loader.beat_data) if self.game_logic.music_sheet_loader.beat_data else 0}")
            
            # 获取最后一个音符的时间
            if self.game_logic.music_sheet_loader.beat_data:
                self.last_note_time = self.game_logic.music_sheet_loader.beat_data[-1].get('time', 0) + 3  # 加3秒额外时间
                self.remaining_notes = len(self.game_logic.music_sheet_loader.beat_data)
        else:
            logger.warning(f"Failed to load music sheet: {self.music_sheet_path}")
    
    def extract_audio(self):
        """
        从视频中提取音频。
        
        Returns:
            bool: 提取成功返回True，否则返回False
        """
        try:
            logger.info(f"Extracting audio for {self.music}")
            
            # 获取视频路径
            video_path = f"example_video/{self.music}.mp4"
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # 创建音频文件
            self.audio_path = f"{self.music}_audio.wav"
            
            # 使用ffmpeg提取音频
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                self.audio_path
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # 执行命令
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 等待进程完成
            stdout, stderr = process.communicate()
            
            # 检查结果
            if process.returncode != 0:
                logger.error(f"Audio extraction failed: {stderr}")
                return False
            
            if not os.path.exists(self.audio_path) or os.path.getsize(self.audio_path) == 0:
                logger.error(f"Output audio file is missing or empty: {self.audio_path}")
                return False
            
            logger.info(f"Audio extracted successfully to {self.audio_path}")
            self.audio_extracted = True
            return True
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def play_audio(self):
        """播放提取的音频"""
        if not self.audio_extracted or not self.audio_path or not os.path.exists(self.audio_path):
            logger.warning("Cannot play audio: audio not extracted or file not found")
            return False
        
        try:
            # 停止当前正在播放的音频
            pygame.mixer.music.stop()
            
            # 加载并播放新音频
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()
            self.audio_playing = True
            logger.info(f"Playing audio: {self.audio_path}")
            return True
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            self.audio_playing = False
            return False
    
    def cleanup_audio(self):
        """清理提取的音频文件"""
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                # 首先停止音乐播放
                pygame.mixer.music.stop()
                self.audio_playing = False
                
                # 确保音频资源完全释放
                pygame.mixer.music.unload()
                
                # 添加小延迟以确保资源释放
                time.sleep(0.1)
                
                # 现在尝试删除文件
                os.remove(self.audio_path)
                logger.info(f"Removed audio file: {self.audio_path}")
                self.audio_path = None
                self.audio_extracted = False
                return True
            except Exception as e:
                logger.error(f"Error cleaning up audio: {e}")
                # 即使删除失败也将状态重置，防止重复尝试删除
                self.audio_playing = False
                self.audio_path = None
                self.audio_extracted = False
                return False
        return True
    
    def process_input(self, events):
        """
        处理输入事件。
        
        Args:
            events: pygame事件列表
            
        Returns:
            bool: 如果需要退出游戏，返回True；否则返回False
        """
        for event in events:
            if event.type == pygame.QUIT:
                self.cleanup_audio()
                return True
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.game_state == PLAYING:
                        self.game_state = PAUSED
                        # 暂停音乐
                        if self.audio_playing:
                            pygame.mixer.music.pause()
                    elif self.game_state == PAUSED:
                        self.game_state = PLAYING
                        # 继续播放音乐
                        if self.audio_playing:
                            pygame.mixer.music.unpause()
                    elif self.game_state == START_SCREEN or self.game_state == SONG_SELECTION:
                        self.cleanup_audio()
                        return True
                elif event.key == pygame.K_p and self.game_state == PLAYING:
                    self.game_state = PAUSED
                    if self.audio_playing:
                        pygame.mixer.music.pause()
                elif event.key == pygame.K_p and self.game_state == PAUSED:
                    self.game_state = PLAYING
                    if self.audio_playing:
                        pygame.mixer.music.unpause()
                elif event.key == pygame.K_q and self.game_state == PAUSED:
                    self.cleanup_audio()
                    return True
                elif event.key == pygame.K_SPACE and self.game_state == PLAYING:
                    self.ui_renderer.toggle_instructions()
                # 测试快捷键，直接开始游戏
                elif event.key == pygame.K_F5:
                    logger.info("Debug shortcut: Starting game directly")
                    if not self.audio_extracted:
                        self.extract_audio()
                    self.countdown_start_time = time.time()
                    self.game_state = COUNTDOWN
                # 添加镜像模式切换按键M
                elif event.key == pygame.K_m:
                    self.ui_renderer.toggle_mirror_mode()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 处理点击事件
                if self.game_state == START_SCREEN:
                    self._handle_difficulty_selection(event.pos)
                elif self.game_state == SONG_SELECTION:
                    self._handle_song_selection(event.pos)
                elif self.game_state == INSTRUCTION_INIT:
                    self._handle_tap_continue()
                elif self.game_state == GAME_OVER:
                    if hasattr(self.ui_renderer, 'restart_button') and self.ui_renderer.restart_button.collidepoint(event.pos):
                        # 重新选择歌曲
                        self.cleanup_audio()
                        self.game_state = SONG_SELECTION
                        logger.info("Restarting game: returning to song selection")
                    elif hasattr(self.ui_renderer, 'exit_button') and self.ui_renderer.exit_button.collidepoint(event.pos):
                        # 退出游戏
                        self.cleanup_audio()
                        return True
        
        return False
    
    def _handle_difficulty_selection(self, mouse_pos):
        """处理难度选择点击"""
        for difficulty, rect in self.ui_renderer.difficulty_buttons.items():
            if rect.collidepoint(mouse_pos):
                logger.info(f"Difficulty selected: {difficulty}")
                self.difficulty = difficulty
                # 如果已经有选择的歌曲，则更新谱面路径
                if self.music in MUSIC_SHEETS:
                    self.music_sheet_path = MUSIC_SHEETS[self.music][difficulty]
                    logger.info(f"Updated music sheet path: {self.music_sheet_path}")
                    
                # 重新创建游戏逻辑实例，确保使用正确的难度和谱面
                self.create_game_logic()
                
                # 提取音频
                if not self.audio_extracted:
                    if not self.extract_audio():
                        logger.error("Failed to extract audio, continuing without audio")
                
                # 开始倒计时
                self.countdown_start_time = time.time()
                self.game_state = INSTRUCTION_INIT
                logger.info(f"State transition: START_SCREEN -> INSTRUCTION_INIT")
                break
    
    def _handle_song_selection(self, mouse_pos):
        """处理歌曲选择点击"""
        for song, rect in self.ui_renderer.song_buttons.items():
            if rect.collidepoint(mouse_pos):
                logger.info(f"Song selected: {song}")
                self.music = song
                
                # 清理之前的音频
                self.cleanup_audio()
                
                # 更新谱面路径
                if self.music in MUSIC_SHEETS:
                    self.music_sheet_path = MUSIC_SHEETS[self.music][self.difficulty]
                    logger.info(f"Updated music sheet path: {self.music_sheet_path}")
                
                # 进入难度选择
                self.game_state = START_SCREEN
                logger.info(f"State transition: SONG_SELECTION -> START_SCREEN")
                break

    def _handle_tap_continue(self):
        """
        Handle tap the screen to continue event.
        Args:
            mouse_pos: Current mouse position
        """
        logger.info("Tap detected: INSTRUCTION_INIT -> INITIALIZATION")
        self.game_state = INITIALIZATION
  
    def update(self, pose_data=None):
        """
        更新游戏状态。
        
        Args:
            pose_data: 来自PoseFramework的姿态数据
        """
        if self.game_state == COUNTDOWN:
            # 倒计时逻辑
            elapsed_countdown_time = time.time() - self.countdown_start_time
            if elapsed_countdown_time > 3.5:  # 给"开始"显示一点时间
                self.game_start_time = time.time()
                self.game_state = PLAYING
                logger.info(f"State transition: COUNTDOWN -> PLAYING")
                
                # 重置游戏逻辑
                self.game_logic.reset()
                self.game_logic.game_start_time = self.game_start_time
                
                # 确保音乐谱面已正确加载
                if not self.game_logic.using_music_sheet:
                    logger.warning("No music sheet loaded, attempting to reload")
                    if self.music in MUSIC_SHEETS:
                        self.music_sheet_path = MUSIC_SHEETS[self.music][self.difficulty]
                        logger.info(f"Reloading music sheet: {self.music_sheet_path}")
                        self.game_logic.music_sheet_path = self.music_sheet_path
                        self.game_logic.using_music_sheet = self.game_logic.music_sheet_loader.load_music_sheet(self.music_sheet_path)
                        if self.game_logic.using_music_sheet:
                            logger.info("Music sheet loaded successfully")
                            if 'video_info' in self.game_logic.music_sheet_loader.json_data:
                                self.game_logic._calculate_mapping_params()
                
                # 开始播放音频
                if self.audio_extracted and not self.audio_playing:
                    self.play_audio()
        
        if self.game_state == INITIALIZATION:
            self.game_logic.check_ready(pose_data)
            if self.game_logic.is_ready:
                logger.info("State transition: INITIALIZATION -> COUNTDOWN")
                self.game_state = COUNTDOWN
                self.countdown_start_time = time.time()

        elif self.game_state == PLAYING and pose_data is not None:
            # 游戏进行中的逻辑
            # 检查命中并更新游戏状态
            self.game_logic.check_hits(pose_data)
            self.game_logic.update_notes()
            
            # 计算经过的时间
            elapsed_time = time.time() - self.game_start_time
            
            # 用于测试的简单音符生成（如果没有音乐谱面）
            if not self.game_logic.using_music_sheet and len(self.game_logic.notes) < 3:
                # 每1.5秒随机生成一个音符
                if len(self.game_logic.notes) == 0 or elapsed_time % 1.5 < 0.1:
                    self.game_logic.spawn_note()
                    logger.info(f"Spawned random note, total notes: {len(self.game_logic.notes)}")
            
            # 更新游戏逻辑
            self.game_logic.update(elapsed_time)
            
            # 检查游戏是否结束
            if self.game_logic.using_music_sheet:
                # 所有音符已生成且没有活跃音符
                if (self.game_logic.current_beat_index >= len(self.game_logic.music_sheet_loader.beat_data) and
                    len(self.game_logic.notes) == 0 and
                    elapsed_time > self.last_note_time):
                    self.game_state = GAME_OVER
                    logger.info(f"State transition: PLAYING -> GAME_OVER")
            elif not pygame.mixer.music.get_busy() and self.audio_playing:
                # 音乐播放完毕
                self.game_state = GAME_OVER
                logger.info(f"State transition: PLAYING -> GAME_OVER (music ended)")
    
    def render(self, screen, camera_frame=None):
        """
        渲染游戏画面。
        
        Args:
            screen: pygame屏幕对象
            camera_frame: 可选的摄像头帧，用于背景显示
        """
        # 处理不同状态的渲染
        if self.game_state == SONG_SELECTION:
            # 渲染歌曲选择界面
            self.ui_renderer.render_song_selection(screen)
        
        elif self.game_state == START_SCREEN:
            # 渲染开始界面（难度选择）
            self.ui_renderer.render_start_screen(screen, self.music)

        elif self.game_state == INSTRUCTION_INIT:
            # instruction to initialization of camera and user pose
            self.ui_renderer.render_instruction_init(screen)
            
        elif self.game_state == INITIALIZATION:
            # initialization of camera and user pose
            if camera_frame is not None:
                # deal with camera frame
                pose_data = self.framework.process_frame(camera_frame)

                # visualize pose
                camera_frame = self.framework.draw_results(camera_frame, pose_data, highlight_palms=True)
                
                # draw instruction
                camera_frame = self.framework.draw_instructions(camera_frame, pose_data)
                # 更新游戏状态
                self.update(pose_data)
            self.ui_renderer.render_initialization(screen, camera_frame)

        elif self.game_state == COUNTDOWN:
            # 渲染倒计时
            countdown_time = 3 - (time.time() - self.countdown_start_time)
            self.ui_renderer.render_countdown(screen, countdown_time, camera_frame)
            
        elif self.game_state == PLAYING:
            # 如果有摄像头帧，处理以获取姿态数据
            pose_data = None
            if camera_frame is not None:
                pose_data = self.framework.process_frame(camera_frame)
                # 绘制姿态可视化
                camera_frame = self.framework.draw_results(camera_frame, pose_data, highlight_palms=True)
                
                # 更新游戏状态
                self.update(pose_data)
            
            # 获取当前游戏状态
            game_state = self.game_logic.get_game_state()
            
            # 添加FPS到游戏状态
            game_state['fps'] = self.current_fps
            
            # 使用UI渲染器绘制游戏
            self.ui_renderer.render_game(screen, game_state, camera_frame)
            
            # 如果需要，绘制指导屏幕
            if self.ui_renderer.should_show_instructions():
                self.ui_renderer.render_instructions(screen)
                
        elif self.game_state == PAUSED:
            # 继续绘制游戏界面，然后覆盖暂停界面
            if self.game_logic:
                game_state = self.game_logic.get_game_state()
                game_state['fps'] = self.current_fps
                self.ui_renderer.render_game(screen, game_state, camera_frame)
            
            # 绘制暂停屏幕
            self.ui_renderer.render_pause_screen(screen)
            
        elif self.game_state == GAME_OVER:
            # 渲染游戏结束画面
            if self.game_logic:
                self.ui_renderer.render_game_over_screen(
                    screen, 
                    self.game_logic.score, 
                    self.game_logic.hits, 
                    self.game_logic.misses, 
                    self.game_logic.max_combo
                )
    
    def run(self):
        """游戏主循环。"""
        self.game_state = SONG_SELECTION  # 先选择歌曲
        self.running = True
        last_time = time.time()
        
        logger.info("Game starting, initial state: SONG_SELECTION")
        
        while self.running:
            # 计算帧率
            current_time = time.time()
            delta_time = current_time - last_time
            self.current_fps = 1.0 / max(0.001, delta_time)  # 防止除以0
            last_time = current_time
            
            # 处理事件
            events = pygame.event.get()
            if self.process_input(events):
                self.running = False
                continue
            
            # 获取摄像头帧
            ret, frame = self.framework.camera.read()
            if not ret:
                logger.error("Failed to read camera frame, exiting game")
                self.running = False
                break
            
            # 如果在COUNTDOWN状态，直接调用update方法检查是否需要转换到PLAYING
            if self.game_state == COUNTDOWN:
                self.update()
            
            # 渲染游戏界面
            self.render(self.screen, frame)
            
            # 刷新显示
            pygame.display.flip()
            
            # 控制帧率
            pygame.time.Clock().tick(60)
        
        # 游戏结束，释放资源
        self.cleanup_audio()
        self.framework.release()
        pygame.quit()