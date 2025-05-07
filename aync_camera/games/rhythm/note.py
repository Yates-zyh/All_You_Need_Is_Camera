"""
音符类模块，用于节奏游戏中的音符表示和管理。
"""
import time
import pygame
import math
from aync_camera.config.game_settings import COLORS, KEYPOINT_LABELS

# 为不同的身体部位定义形状类型
SHAPE_MAPPING = {
    # 头部 - 圆形
    0: "circle",  # 鼻子
    
    # 手臂 - 三角形
    5: "triangle",  # 左肩
    6: "triangle",  # 右肩
    7: "triangle",  # 左肘
    8: "triangle",  # 右肘
    9: "triangle",  # 左手腕
    10: "triangle",  # 右手腕
    
    # 躯干 - 菱形
    11: "diamond",  # 左髋
    12: "diamond",  # 右髋
    
    # 腿部 - 方形
    13: "square",  # 左膝
    14: "square",  # 右膝
    15: "square",  # 左踝
    16: "square",  # 右踝
}

# 为左右两侧定义不同的颜色
SIDE_COLORS = {
    "left": COLORS["cyan"],     # 左侧 - 青色
    "right": COLORS["purple"],  # 右侧 - 紫色
    "center": COLORS["yellow"], # 中间 - 黄色
}

# 判定等级及其对应的颜色
JUDGMENT_COLORS = {
    "Perfect": COLORS["gold"],    # 完美 - 金色
    "Good": COLORS["green"],      # 良好 - 绿色
    "Miss": COLORS["red"],        # 错过 - 红色
}

class Note:
    """表示节奏游戏中带有收缩判定圈的音符。"""
    
    def __init__(self, x: int, y: int, radius: int, max_radius: int, target_time: float, duration: float = 2.0, 
                 keypoint_name: str = None, keypoint_index: int = None, hit_window_width: float = 0.4, perfect_threshold: float = 0.08):
        """
        初始化一个新的音符。
        
        Args:
            x: 音符中心的X坐标
            y: 音符中心的Y坐标
            radius: 音符本身的半径（内圈）
            max_radius: 判定圈的初始半径（外圈）
            target_time: 音符应该被精确命中的绝对游戏时间点
            duration: 音符从出现到达到target_time所需的时间（判定圈从最大缩到最小所需的时间）
            keypoint_name: 这个音符对应的身体部位名称（例如，"Left Wrist"）
            keypoint_index: 这个音符对应的关键点索引（例如，9表示左手腕）
            hit_window_width: 有效命中窗口的总宽度（秒）
            perfect_threshold: Perfect判定所需的时间精度阈值（秒）
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.max_radius = max_radius
        self.current_radius = max_radius
        self.duration = duration
        
        # 新的时间属性
        self.target_time = target_time  # 音符应该被命中的精确时间点
        self.appear_time = target_time - duration  # 音符首次出现的时间点
        
        # 判定窗口
        self.hit_window_width = hit_window_width
        self.hit_window_start = target_time - hit_window_width / 2
        self.hit_window_end = target_time + hit_window_width / 2
        self.perfect_threshold = perfect_threshold
        
        # 状态属性
        self.hit = False
        self.missed = False
        self.judged = False  # 表示这个音符已经被判定
        self.judgment = None  # 用于存储判定结果："Perfect", "Good", "Miss"
        self.hit_time = None  # 记录精确命中时间
        self.hit_feedback_time = None
        
        # 添加渐隐效果相关属性
        self.fade_out = False
        self.fade_start_time = None
        self.fade_duration = 0.5  # 消失动画持续0.5秒
        self.opacity = 255  # 完全不透明
        
        # 添加身体部位名称和索引
        self.keypoint_name = keypoint_name
        self.keypoint_index = keypoint_index
        
        # 新增：根据keypoint_index确定形状和颜色
        self.shape_type = "circle"  # 默认为圆形
        if keypoint_index is not None and keypoint_index in SHAPE_MAPPING:
            self.shape_type = SHAPE_MAPPING[keypoint_index]
        
        # 确定颜色（基于左右侧）
        self.color = COLORS["blue"]  # 默认颜色
        if keypoint_name:
            if "left" in keypoint_name.lower():
                self.color = SIDE_COLORS["left"]
            elif "right" in keypoint_name.lower():
                self.color = SIDE_COLORS["right"]
            else:
                self.color = SIDE_COLORS["center"]
        
    def update(self):
        """
        根据当前时间更新音符状态。
        
        Returns:
            如果音符应该被移除，则返回True，否则返回False
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
            
        # 计算音符从出现时间到当前时间的进度
        elapsed_visual_time = current_time - self.appear_time
        progress = min(1.0, max(0.0, elapsed_visual_time / self.duration))
        
        # 判定圈从max_radius收缩到note半径
        self.current_radius = self.radius + (self.max_radius - self.radius) * (1.0 - progress)
        
        # 检查是否超过判定窗口结束时间（Miss判定）
        if current_time > self.hit_window_end and not self.judged:
            self.missed = True
            self.judged = True
            self.judgment = "Miss"
            return False
        
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
            color = self.color
            if self.judgment:
                color = JUDGMENT_COLORS.get(self.judgment, self.color)
                    
            # 创建带透明度的表面
            s_size = self.radius * 2
            note_surface = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
            border_surface = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
            
            # 根据形状类型绘制不同的几何形状
            self._draw_shape(note_surface, (*color[:3], self.opacity), (self.radius, self.radius), self.radius, fill=True)
            self._draw_shape(border_surface, (*COLORS["white"][:3], self.opacity), (self.radius, self.radius), self.radius, fill=False)
            
            # 绘制到屏幕上
            screen.blit(note_surface, (self.x - self.radius, self.y - self.radius))
            screen.blit(border_surface, (self.x - self.radius, self.y - self.radius))
            
            # 绘制判定结果（如果有）
            if self.judgment:
                font = pygame.font.Font(None, 30)  # 使用稍大号字体
                judgment_text = font.render(self.judgment, True, JUDGMENT_COLORS.get(self.judgment, COLORS["white"]))
                judgment_rect = judgment_text.get_rect(center=(self.x, self.y - self.radius - 20))
                # 创建文本表面并绘制
                judgment_surface = pygame.Surface(judgment_text.get_size(), pygame.SRCALPHA)
                judgment_surface.fill((0, 0, 0, 0))  # 透明背景
                judgment_surface.blit(judgment_text, (0, 0))
                screen.blit(judgment_surface, judgment_rect)
            
            # 绘制部位名称（如果有）
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
        
        # 绘制音符（内圈）
        color = self.color  # 使用动态颜色
        if self.judgment:
            color = JUDGMENT_COLORS.get(self.judgment, self.color)
            
        # 根据形状类型绘制不同的几何形状
        self._draw_shape(screen, color, (self.x, self.y), self.radius, fill=True)
        self._draw_shape(screen, COLORS["white"], (self.x, self.y), self.radius, fill=False)
        
        # 绘制判定结果（如果已判定）
        if self.judgment:
            font = pygame.font.Font(None, 30)  # 使用稍大号字体
            judgment_text = font.render(self.judgment, True, JUDGMENT_COLORS.get(self.judgment, COLORS["white"]))
            judgment_rect = judgment_text.get_rect(center=(self.x, self.y - self.radius - 20))
            screen.blit(judgment_text, judgment_rect)
        
        # 绘制部位名称（如果有）
        if self.keypoint_name:
            font = pygame.font.Font(None, 20)  # 使用小号字体
            # 转换部位名称为显示格式 (例如: "left_wrist" -> "Left Wrist")
            display_name = self.keypoint_name.replace("_", " ").title()
            text = font.render(display_name, True, COLORS["white"])
            text_rect = text.get_rect(center=(self.x, self.y + self.radius + 10))
            screen.blit(text, text_rect)
            
    def _draw_shape(self, surface, color, center, radius, fill=True):
        """
        根据形状类型绘制不同的几何形状。
        
        Args:
            surface: 要绘制的Pygame表面
            color: 形状的颜色
            center: 形状的中心坐标 (x, y)
            radius: 形状的半径或大小参数
            fill: 是否填充形状
        """
        x, y = center
        line_width = 0 if fill else 2  # 0表示填充，非0表示线宽
        
        if self.shape_type == "circle":
            # 圆形 (头部)
            pygame.draw.circle(surface, color, (x, y), radius, line_width)
            
        elif self.shape_type == "square":
            # 方形 (腿部)
            rect_size = radius * 1.8  # 方形略大于圆形
            rect = pygame.Rect(x - rect_size/2, y - rect_size/2, rect_size, rect_size)
            pygame.draw.rect(surface, color, rect, line_width)
            
        elif self.shape_type == "triangle":
            # 三角形 (手臂)
            # 计算三角形的三个顶点
            top = (x, y - radius)
            left = (x - radius, y + radius)
            right = (x + radius, y + radius)
            pygame.draw.polygon(surface, color, [top, left, right], line_width)
            
        elif self.shape_type == "diamond":
            # 菱形 (躯干)
            # 计算菱形的四个顶点
            top = (x, y - radius)
            right = (x + radius, y)
            bottom = (x, y + radius)
            left = (x - radius, y)
            pygame.draw.polygon(surface, color, [top, right, bottom, left], line_width)