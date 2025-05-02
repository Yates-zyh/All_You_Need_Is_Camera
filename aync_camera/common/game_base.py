"""
游戏基类模块，为所有游戏类型定义统一接口。
"""
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import pygame

from aync_camera.core.pose_framework import PoseFramework


class GameBase:
    """所有游戏实现的基类，定义通用接口和共享功能。"""
    
    def __init__(
        self, 
        camera_id: int = 0, 
        model_path: str = "yolo11x-pose.pt",
        screen_width: int = 1280,
        screen_height: int = 720
    ):
        """
        初始化游戏基类。
        
        Args:
            camera_id: 摄像头设备ID
            model_path: YOLOv8-Pose模型路径
            screen_width: 游戏窗口宽度
            screen_height: 游戏窗口高度
        """
        # 初始化pygame
        pygame.init()
        
        # 设置屏幕尺寸
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # 创建时钟对象，用于控制帧率
        self.clock = pygame.time.Clock()
        
        # 游戏状态
        self.running = True
        self.paused = False
        self.game_start_time = time.time()
        
        # 初始化姿态识别框架
        self.framework = PoseFramework(
            model_path=model_path, 
            confidence_threshold=0.5  # 默认置信度阈值，子类可以覆盖
        )
        self.cap = self.framework.setup_camera(camera_id=camera_id)
        
        # 性能跟踪
        self.fps_history = []
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
    def initialize(self):
        """
        初始化游戏资源和状态。
        子类必须实现此方法。
        """
        raise NotImplementedError
        
    def process_input(self, events):
        """
        处理输入事件。
        子类必须实现此方法。
        
        Args:
            events: pygame事件列表
            
        Returns:
            bool: 如果需要退出游戏，返回True；否则返回False
        """
        raise NotImplementedError
    
    def update(self, pose_data):
        """
        更新游戏状态。
        子类必须实现此方法。
        
        Args:
            pose_data: 来自PoseFramework的姿态数据
        """
        raise NotImplementedError
    
    def render(self, screen, camera_frame=None):
        """
        渲染游戏画面。
        子类必须实现此方法。
        
        Args:
            screen: pygame屏幕对象
            camera_frame: 可选的摄像头帧，用于背景显示
        """
        raise NotImplementedError
    
    def update_fps(self):
        """更新并计算当前FPS。"""
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_fps_update)
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 60:  # 保留最近60秒的FPS历史
                self.fps_history.pop(0)
            self.frame_count = 0
            self.last_fps_update = now
            
    def run(self):
        """
        运行游戏主循环。
        这是一个通用实现，子类可以根据需要覆盖。
        """
        try:
            while self.running:
                # 处理事件
                events = pygame.event.get()
                quit_requested = self.process_input(events)
                if quit_requested:
                    self.running = False
                    break
                
                # 捕获摄像头帧
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # 处理帧并获取姿态数据
                pose_data = self.framework.process_frame(frame)
                
                # 更新游戏状态（如果没有暂停）
                if not self.paused:
                    self.update(pose_data)
                
                # 渲染游戏
                self.render(self.screen, camera_frame=frame)
                
                # 更新显示
                pygame.display.flip()
                
                # 更新FPS
                self.update_fps()
                
                # 限制帧率
                self.clock.tick(60)
                
        except Exception as e:
            print(f"Error in game loop: {str(e)}")
        finally:
            # 清理资源
            pygame.quit()
            self.cap.release()
            cv2.destroyAllWindows()
    
    def quit(self):
        """退出游戏。"""
        self.running = False