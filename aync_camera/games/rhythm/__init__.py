"""
节奏游戏主模块。
"""
import time
import pygame

from aync_camera.common.game_base import GameBase
from aync_camera.games.rhythm.game_logic import RhythmGameLogic
from aync_camera.ui.rhythm.renderer import RhythmGameRenderer
from aync_camera.config.rhythm_config import MUSIC_SHEETS, DEFAULT_MUSIC


class RhythmGame(GameBase):
    """Cytus风格的节奏游戏，使用身体姿态作为控制。"""
    
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
            model_path: YOLOv8-Pose模型路径
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
        pygame.display.set_caption(f"Cytus-style Rhythm Game - {difficulty.capitalize()} Mode")
        
        # 确定音乐谱面路径
        if not music_sheet_path and music and music in MUSIC_SHEETS:
            music_sheet_path = MUSIC_SHEETS[music][difficulty]
        elif not music_sheet_path and not music:
            # 使用默认音乐
            music = DEFAULT_MUSIC
            music_sheet_path = MUSIC_SHEETS[music][difficulty]
        
        # 创建游戏逻辑实例
        self.game_logic = RhythmGameLogic(
            difficulty=difficulty,
            screen_width=screen_width,
            screen_height=screen_height,
            music=music,
            music_sheet_path=music_sheet_path
        )
        
        # 创建UI渲染器实例
        self.ui_renderer = RhythmGameRenderer(
            screen_width=screen_width,
            screen_height=screen_height
        )
    
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
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.paused:
                        self.paused = False
                    else:
                        self.paused = True
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_q and self.paused:
                    return True
                elif event.key == pygame.K_SPACE:
                    self.ui_renderer.toggle_instructions()
        
        return False
    
    def update(self, pose_data):
        """
        更新游戏状态。
        
        Args:
            pose_data: 来自PoseFramework的姿态数据
        """
        # 检查命中并更新游戏状态
        self.game_logic.check_hits(pose_data)
        self.game_logic.update_notes()
        
        # 计算经过的时间
        elapsed_time = time.time() - self.game_start_time
        
        # 更新游戏逻辑
        self.game_logic.update(elapsed_time)
    
    def render(self, screen, camera_frame=None):
        """
        渲染游戏画面。
        
        Args:
            screen: pygame屏幕对象
            camera_frame: 可选的摄像头帧，用于背景显示
        """
        # 如果有摄像头帧，处理以获取姿态数据
        if camera_frame is not None:
            pose_data = self.framework.process_frame(camera_frame)
            # 绘制姿态可视化
            camera_frame = self.framework.draw_results(camera_frame, pose_data, highlight_palms=True)
        
        # 获取当前游戏状态
        game_state = self.game_logic.get_game_state()
        
        # 添加FPS到游戏状态
        game_state['fps'] = self.current_fps
        
        # 使用UI渲染器绘制游戏
        self.ui_renderer.render_game(screen, game_state, camera_frame)
        
        # 如果需要，绘制指导或暂停屏幕
        if self.ui_renderer.should_show_instructions():
            self.ui_renderer.render_instructions(screen)
        
        if self.paused:
            self.ui_renderer.render_pause_screen(screen)