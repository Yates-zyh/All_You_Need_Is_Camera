"""
节奏游戏UI渲染模块，负责游戏界面的绘制。
"""
import time
import cv2
import pygame
import numpy as np

from aync_camera.config.game_settings import COLORS

class RhythmGameRenderer:
    """节奏游戏界面渲染器，处理所有UI元素的绘制。"""
    
    def __init__(self, screen_width, screen_height):
        """
        初始化UI渲染器。
        
        Args:
            screen_width: 游戏窗口宽度
            screen_height: 游戏窗口高度
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 加载字体
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # 指令覆盖标志
        self.show_instructions = True
        self.instructions_timer = time.time() + 5.0  # 显示5秒
    
    def render_game(self, screen, game_state, camera_frame=None):
        """
        渲染完整的游戏画面。
        
        Args:
            screen: Pygame屏幕对象
            game_state: 游戏状态字典
            camera_frame: 可选的摄像头帧，用于背景显示
        """
        # 清空屏幕
        screen.fill(COLORS["black"])
        
        # 如果提供了摄像头帧，将其作为背景绘制
        if camera_frame is not None:
            # 调整大小以适应屏幕
            camera_frame = cv2.resize(camera_frame, (self.screen_width, self.screen_height))
            # 转换为RGB用于pygame
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            # 转换为pygame surface
            surface = pygame.surfarray.make_surface(camera_frame.swapaxes(0, 1))
            # 添加透明度
            surface.set_alpha(160)  # 稍微更不透明，以便更好地看到姿态
            # 绘制
            screen.blit(surface, (0, 0))
        
        # 绘制音符
        for note in game_state['notes']:
            note.draw(screen)
        
        # 绘制命中反馈动画
        self.render_hit_feedback(screen, game_state['hit_feedback'], game_state['hit_feedback_duration'])
        
        # 绘制分数和连击
        self.render_score(screen, game_state['score'], game_state['combo'])
        
        # 绘制统计信息
        self.render_stats(screen, game_state['hits'], game_state['misses'], game_state['max_combo'], game_state['difficulty'])
        
        # 绘制FPS（如果提供）
        if 'fps' in game_state:
            self.render_fps(screen, game_state['fps'])
    
    def render_score(self, screen, score, combo):
        """
        渲染分数和连击。
        
        Args:
            screen: Pygame屏幕对象
            score: 当前分数
            combo: 当前连击
        """
        # 绘制分数
        score_text = self.font_medium.render(f"Score: {score}", True, COLORS["white"])
        screen.blit(score_text, (10, 10))
        
        # 绘制连击
        combo_text = self.font_medium.render(f"Combo: {combo}", True, COLORS["yellow"])
        screen.blit(combo_text, (10, 50))
    
    def render_stats(self, screen, hits, misses, max_combo, difficulty):
        """
        渲染游戏统计信息。
        
        Args:
            screen: Pygame屏幕对象
            hits: 成功命中次数
            misses: 错过次数
            max_combo: 最大连击
            difficulty: 游戏难度
        """
        # 绘制统计信息
        stats_text = self.font_small.render(
            f"Hits: {hits}  Misses: {misses}  Max Combo: {max_combo}",
            True,
            COLORS["white"]
        )
        screen.blit(stats_text, (10, 90))
        
        # 绘制难度
        diff_text = self.font_small.render(
            f"Difficulty: {difficulty.capitalize()}",
            True,
            COLORS["cyan"]
        )
        screen.blit(diff_text, (10, 120))
    
    def render_fps(self, screen, fps):
        """
        渲染FPS信息。
        
        Args:
            screen: Pygame屏幕对象
            fps: 当前FPS值
        """
        fps_text = self.font_small.render(f"FPS: {fps:.1f}", True, COLORS["white"])
        screen.blit(fps_text, (self.screen_width - fps_text.get_width() - 10, 10))
    
    def render_hit_feedback(self, screen, hit_feedback, hit_feedback_duration):
        """
        渲染命中反馈动画。
        
        Args:
            screen: Pygame屏幕对象
            hit_feedback: 命中反馈数据列表
            hit_feedback_duration: 命中反馈持续时间
        """
        current_time = time.time()
        
        for hit_x, hit_y, hit_time, hit_type in hit_feedback:
            # 计算动画进度（0.0到1.0）
            elapsed = current_time - hit_time
            if elapsed > hit_feedback_duration:
                continue
                
            progress = elapsed / hit_feedback_duration
            size = int(50 * (1.0 - progress))  # 开始大，随着动画进展而缩小
            alpha = int(255 * (1.0 - progress))  # 淡出
            
            # 绘制命中效果
            color = COLORS["green"] if hit_type == "hit" else COLORS["red"]
            hit_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(hit_surface, (*color, alpha), (size, size), size)
            screen.blit(hit_surface, (hit_x - size, hit_y - size))
            
            # 绘制"HIT!"文字（向上飘动）
            text = self.font_medium.render("HIT!", True, color)
            text_surface = pygame.Surface(text.get_size(), pygame.SRCALPHA)
            text_surface.fill((0, 0, 0, 0))  # 透明
            text_surface.blit(text, (0, 0))
            text_surface.set_alpha(alpha)
            offset_y = int(30 * progress)  # 文字向上飘动
            screen.blit(text_surface, (hit_x - text.get_width() // 2, hit_y - text.get_height() - offset_y))
    
    def render_instructions(self, screen):
        """
        绘制游戏指导覆盖层。
        
        Args:
            screen: Pygame屏幕对象
        """
        # 半透明覆盖层
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # 带透明度的黑色
        screen.blit(overlay, (0, 0))
        
        # 标题
        title = self.font_large.render("Cytus-style Rhythm Game", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        screen.blit(title, title_rect)
        
        # 指导
        instructions = [
            "Move your hands to hit the notes when their circle shrinks to minimum",
            "Each note has a judgment circle that shrinks over time",
            "When the circle is at its smallest, move your hand to the note",
            "Both Left and Right hands can be used to hit any note",
            "",
            "Press ESC to pause or quit",
            "Press Space to toggle instructions"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_medium.render(line, True, COLORS["white"])
            rect = text.get_rect(center=(self.screen_width // 2, 180 + i * 40))
            screen.blit(text, rect)
    
    def render_pause_screen(self, screen):
        """
        绘制暂停屏幕覆盖层。
        
        Args:
            screen: Pygame屏幕对象
        """
        # 半透明覆盖层
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # 带透明度的黑色
        screen.blit(overlay, (0, 0))
        
        # 标题
        title = self.font_large.render("PAUSED", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(title, title_rect)
        
        # 指导
        instructions = [
            "Press P or ESC to resume",
            "Press Q to quit"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_medium.render(line, True, COLORS["white"])
            rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 20 + i * 40))
            screen.blit(text, rect)
            
    def should_show_instructions(self):
        """
        检查是否应该显示指导。
        
        Returns:
            bool: 如果应该显示指导则返回True，否则返回False
        """
        if self.show_instructions and time.time() > self.instructions_timer:
            self.show_instructions = False
        
        return self.show_instructions
    
    def toggle_instructions(self):
        """切换指导显示状态。"""
        self.show_instructions = not self.show_instructions
        if self.show_instructions:
            self.instructions_timer = time.time() + 5.0