"""
音符类模块，用于节奏游戏中的音符表示和管理。
"""
import time
import pygame
from aync_camera.config.game_settings import COLORS

class Note:
    """表示节奏游戏中带有收缩判定圈的音符。"""
    
    def __init__(self, x: int, y: int, radius: int, max_radius: int, duration: float = 2.0, keypoint_name: str = None, keypoint_index: int = None):
        """
        初始化一个新的音符。
        
        Args:
            x: 音符中心的X坐标
            y: 音符中心的Y坐标
            radius: 音符本身的半径（内圈）
            max_radius: 判定圈的初始半径（外圈）
            duration: 判定圈完全收缩所需的时间（秒）
            keypoint_name: 这个音符对应的身体部位名称（例如，"Left Wrist"）
            keypoint_index: 这个音符对应的关键点索引（例如，9表示左手腕）
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.max_radius = max_radius
        self.current_radius = max_radius
        self.duration = duration
        self.start_time = time.time()
        self.hit = False
        self.missed = False
        self.judged = False  # 表示这个音符已经被判定
        self.hit_feedback_time = None
        # 添加渐隐效果相关属性
        self.fade_out = False
        self.fade_start_time = None
        self.fade_duration = 0.5  # 消失动画持续0.5秒 (原值为1.0秒)
        self.opacity = 255  # 完全不透明
        # 添加身体部位名称和索引
        self.keypoint_name = keypoint_name
        self.keypoint_index = keypoint_index
        
    def update(self):
        """
        根据经过的时间更新判定圈半径。
        
        Returns:
            如果判定时机已到（圈完全收缩）则返回True，否则返回False
        """
        current_time = time.time()
        
        # 处理淡出效果
        if self.fade_out:
            fade_elapsed = current_time - self.fade_start_time
            # 计算不透明度 (从255到0)
            self.opacity = max(0, 255 - int(255 * (fade_elapsed / self.fade_duration)))
            # 如果完全透明，可以移除
            if self.opacity <= 0:
                return True
            return False
            
        # 如果已判定但还没开始淡出，启动淡出倒计时
        if self.judged and not self.fade_out:
            self.fade_out = True
            self.fade_start_time = current_time
            return False
        
        # 正常更新逻辑（圈的缩小）
        if self.judged:
            return False
            
        elapsed = current_time - self.start_time
        progress = min(1.0, elapsed / self.duration)
        
        # 判定圈从max_radius收缩到note半径
        self.current_radius = self.max_radius - progress * (self.max_radius - self.radius)
        
        # 检查判定圈是否完全收缩（判定时机）
        if progress >= 1.0:
            return True
            
        return False
    
    def draw(self, screen):
        """
        绘制音符和判定圈。
        
        Args:
            screen: 要绘制的Pygame屏幕
        """
        # 如果在淡出状态，绘制半透明版本
        if self.fade_out:
            # 绘制半透明的note
            color = COLORS["blue"]
            if self.hit:
                color = COLORS["green"]
            elif self.missed:
                color = COLORS["red"]
                
            # 创建带透明度的表面
            note_surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            border_surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            
            # 绘制带透明度的内部圆
            pygame.draw.circle(note_surface, (*color[:3], self.opacity), (self.radius, self.radius), self.radius)
            # 绘制带透明度的边框
            pygame.draw.circle(border_surface, (*COLORS["white"][:3], self.opacity), (self.radius, self.radius), self.radius, 2)
            
            # 绘制到屏幕上
            screen.blit(note_surface, (self.x - self.radius, self.y - self.radius))
            screen.blit(border_surface, (self.x - self.radius, self.y - self.radius))
            
            # 绘制部位名称（如果有），确保使用英文
            if self.keypoint_name:
                font = pygame.font.Font(None, 20)  # 使用小号字体
                # 转换部位名称为显示格式 (例如: "left_wrist" -> "Left Wrist")
                display_name = self.keypoint_name.replace("_", " ").title()
                text = font.render(display_name, True, (*COLORS["white"][:3], self.opacity))
                text_rect = text.get_rect(center=(self.radius, self.radius))
                # 创建文本表面
                text_surface = pygame.Surface(text.get_size(), pygame.SRCALPHA)
                text_surface.fill((0, 0, 0, 0))  # 透明背景
                text_surface.blit(text, (0, 0))
                # 绘制文本
                screen.blit(text_surface, (self.x - text_rect.width // 2, self.y + self.radius + 5))
            
            return
        
        # 正常绘制逻辑
        # 绘制判定圈（外圈）
        if not self.judged:
            # 绘制外圈（判定圈）
            pygame.draw.circle(screen, COLORS["white"], (self.x, self.y), int(self.current_radius), 2)
        
        # 绘制音符圆圈（内圈）
        color = COLORS["blue"]  # 普通音符
        if self.hit:
            color = COLORS["green"]  # 命中的音符
        elif self.missed:
            color = COLORS["red"]  # 错过的音符
            
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)
        # 添加音符圆圈的边框
        pygame.draw.circle(screen, COLORS["white"], (self.x, self.y), self.radius, 2)
        
        # 绘制部位名称（如果有），确保使用英文
        if self.keypoint_name:
            font = pygame.font.Font(None, 20)  # 使用小号字体
            # 转换部位名称为显示格式 (例如: "left_wrist" -> "Left Wrist")
            display_name = self.keypoint_name.replace("_", " ").title()
            text = font.render(display_name, True, COLORS["white"])
            text_rect = text.get_rect(center=(self.x, self.y + self.radius + 10))
            screen.blit(text, text_rect)