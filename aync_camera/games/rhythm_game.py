"""
Cytus-style Rhythm Game implementation using the PoseFramework.
"""
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pygame

from aync_camera.core.pose_framework import PoseFramework
from aync_camera.games.game_config import (COLORS, COMBO_MULTIPLIER,
                                          DEFAULT_LANE_KEYPOINT_MAP,
                                          DEFAULT_SCREEN_HEIGHT,
                                          DEFAULT_SCREEN_WIDTH,
                                          DIFFICULTY_PRESETS, KEYPOINT_LABELS,
                                          SCORE_PER_HIT)


class Note:
    """Represents a note with shrinking judgment circle in the rhythm game."""
    
    def __init__(self, x: int, y: int, radius: int, max_radius: int, duration: float = 2.0):
        """
        Initialize a new note.
        
        Args:
            x: X-coordinate of note center
            y: Y-coordinate of note center
            radius: Radius of the note itself (inner circle)
            max_radius: Initial radius of the judgment circle (outer circle)
            duration: Time in seconds for judgment circle to shrink completely
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
        self.judged = False  # Flag to indicate this note has been judged
        self.hit_feedback_time = None
        # 添加消除相关属性
        self.fade_out = False
        self.fade_start_time = None
        self.fade_duration = 1.0  # 消失动画持续1秒
        self.opacity = 255  # 完全不透明
        
    def update(self):
        """
        Update judgment circle radius based on elapsed time.
        
        Returns:
            True if judgment timing is reached (circle fully shrunk), False otherwise
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
        
        # Shrink judgment circle from max_radius to note radius
        self.current_radius = self.max_radius - progress * (self.max_radius - self.radius)
        
        # Check if judgment circle is fully shrunk (judgment timing)
        if progress >= 1.0:
            return True
            
        return False
    
    def draw(self, screen):
        """
        Draw the note and judgment circle.
        
        Args:
            screen: Pygame screen to draw on
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
            return
        
        # 正常绘制逻辑
        # Draw judgment circle (outer circle)
        if not self.judged:
            # Draw outer circle (judgment circle)
            pygame.draw.circle(screen, COLORS["white"], (self.x, self.y), int(self.current_radius), 2)
        
        # Draw note circle (inner circle)
        color = COLORS["blue"]  # Regular note
        if self.hit:
            color = COLORS["green"]  # Hit note
        elif self.missed:
            color = COLORS["red"]  # Missed note
            
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)
        # Add border to note circle
        pygame.draw.circle(screen, COLORS["white"], (self.x, self.y), self.radius, 2)


class CytusStyleGame:
    """Cytus-style Rhythm Game using body pose as controls."""
    
    def __init__(
        self, 
        camera_id: int = 0, 
        model_path: str = "yolov8n-pose.pt",
        difficulty: str = "normal",
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT
    ):
        """
        Initialize the game.
        
        Args:
            camera_id: Camera device ID
            model_path: Path to YOLOv8-Pose model
            difficulty: Game difficulty ('easy', 'normal', 'hard')
            screen_width: Width of the game window
            screen_height: Height of the game window
        """
        # Initialize pygame
        pygame.init()
        
        # Load difficulty settings
        self.difficulty = difficulty
        self.difficulty_settings = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])
        
        # Game settings
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.note_radius = 40  # Inner circle radius
        self.max_radius = 120  # Initial judgment circle radius
        self.acceptance_radius = 70  # How close hand must be to note center to count as hit
        self.spawn_rate = self.difficulty_settings["spawn_rate"] * 2  # Adjusted for new gameplay
        
        # 增加note持续时间，减慢圈的缩小速度（原来是5.0减去速度，现在是8.0减去速度）
        self.note_duration = max(3.0, 8.0 - self.difficulty_settings["note_speed"])  
        
        self.confidence_threshold = self.difficulty_settings["confidence_threshold"]
        self.padding = 100  # Padding from screen edges for note spawning
        
        # Game state
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.game_over = False
        self.paused = False
        
        # Active notes
        self.notes = []
        
        # Hit feedback (visual indicator for successful hits)
        self.hit_feedback = []  # List of (x, y, time, hit_type) tuples for hit animations
        self.hit_feedback_duration = 0.5  # Duration in seconds to show hit feedback
        
        # Setup screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Cytus-style Rhythm Game - {difficulty.capitalize()} Mode")
        self.clock = pygame.time.Clock()
        
        # Initialize pose framework
        self.framework = PoseFramework(
            model_path=model_path, 
            confidence_threshold=self.confidence_threshold
        )
        self.cap = self.framework.setup_camera(camera_id=camera_id)
        
        # Keypoint indices (9: left wrist, 10: right wrist)
        self.keypoint_indices = [9, 10]
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state
        self.running = True
        self.game_start_time = time.time()
        
        # Performance tracking
        self.fps_history = []
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
        # Instructions overlay flag
        self.show_instructions = True
        self.instructions_timer = time.time() + 5.0  # Show for 5 seconds

    def check_hits(self, pose_data):
        """
        Check for hits between player pose and notes.
        
        Args:
            pose_data: Pose data from PoseFramework
        """
        if not pose_data['persons']:
            return
            
        # Get the first detected person
        person = pose_data['persons'][0]
        keypoints = person['keypoints_xy']
        keypoints_conf = person['keypoints_conf']
        
        # Only consider wrist keypoints (9: left wrist, 10: right wrist)
        wrist_indices = [9, 10]
        palm_positions = {}
        
        # Get palm positions with confidence threshold
        for idx in wrist_indices:
            if idx < len(keypoints) and keypoints_conf[idx] >= self.confidence_threshold:
                palm_positions[idx] = (int(keypoints[idx][0]), int(keypoints[idx][1]))
        
        # Check each note
        for note in self.notes:
            if note.judged:
                continue
            
            # Check for note collision with palms
            for idx in wrist_indices:
                if idx not in palm_positions:
                    continue
                
                # Get palm position
                px, py = palm_positions[idx]
                
                # Calculate distance from palm to note center
                distance = ((note.x - px) ** 2 + (note.y - py) ** 2) ** 0.5
                
                # Check if palm is within acceptance radius of note center
                if distance <= self.acceptance_radius:
                    # Hit!
                    self.hits += 1
                    
                    # Calculate score (with combo multiplier if enabled)
                    hit_score = SCORE_PER_HIT
                    if COMBO_MULTIPLIER:
                        hit_score *= max(1, self.combo)
                    self.score += hit_score
                    
                    # Update combo
                    self.combo += 1
                    self.max_combo = max(self.max_combo, self.combo)
                    
                    # Mark note as hit
                    note.hit = True
                    note.judged = True  # Mark this note as judged
                    
                    # Add visual hit feedback
                    note.hit_feedback_time = time.time()
                    self.hit_feedback.append((note.x, note.y, note.hit_feedback_time, "hit"))
                    break
                
            # Check if note is missed (judgment timing reached without being hit)
            if not note.hit and note.update():
                # Note was missed
                self.misses += 1
                self.combo = 0
                note.missed = True
                note.judged = True  # Mark this note as judged

    def draw_game(self, camera_frame=None):
        """
        Draw the game screen.
        
        Args:
            camera_frame: Optional camera frame to show in background
        """
        # Clear screen
        self.screen.fill(COLORS["black"])
        
        # Draw camera frame as background if provided
        if camera_frame is not None:
            # Resize to fit screen
            camera_frame = cv2.resize(camera_frame, (self.screen_width, self.screen_height))
            # Process frame through pose framework to get pose data
            pose_data = self.framework.process_frame(camera_frame)
            # Draw pose visualization
            camera_frame_with_pose = self.framework.draw_results(camera_frame, pose_data, highlight_palms=True)
            # Convert to RGB for pygame
            camera_frame_with_pose = cv2.cvtColor(camera_frame_with_pose, cv2.COLOR_BGR2RGB)
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(camera_frame_with_pose.swapaxes(0, 1))
            # Add transparency
            surface.set_alpha(160)  # Slightly more opaque to see the pose better
            # Draw
            self.screen.blit(surface, (0, 0))
        
        # Draw notes
        for note in self.notes:
            note.draw(self.screen)
        
        # Draw hit feedback animations
        current_time = time.time()
        for hit_x, hit_y, hit_time, hit_type in self.hit_feedback[:]:
            # Calculate animation progress (0.0 to 1.0)
            elapsed = current_time - hit_time
            if elapsed > self.hit_feedback_duration:
                self.hit_feedback.remove((hit_x, hit_y, hit_time, hit_type))
                continue
                
            progress = elapsed / self.hit_feedback_duration
            size = int(50 * (1.0 - progress))  # Start big, shrink as animation progresses
            alpha = int(255 * (1.0 - progress))  # Fade out
            
            # Draw hit effect
            color = COLORS["green"] if hit_type == "hit" else COLORS["red"]
            hit_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(hit_surface, (*color, alpha), (size, size), size)
            self.screen.blit(hit_surface, (hit_x - size, hit_y - size))
            
            # Draw "HIT!" text that floats upward
            text = self.font_medium.render("HIT!", True, color)
            text_surface = pygame.Surface(text.get_size(), pygame.SRCALPHA)
            text_surface.fill((0, 0, 0, 0))  # Transparent
            text_surface.blit(text, (0, 0))
            text_surface.set_alpha(alpha)
            offset_y = int(30 * progress)  # Text floats upward
            self.screen.blit(text_surface, (hit_x - text.get_width() // 2, hit_y - text.get_height() - offset_y))
            
        # Draw score and combo
        score_text = self.font_medium.render(f"Score: {self.score}", True, COLORS["white"])
        self.screen.blit(score_text, (10, 10))
        
        combo_text = self.font_medium.render(f"Combo: {self.combo}", True, COLORS["yellow"])
        self.screen.blit(combo_text, (10, 50))
        
        # Draw stats
        stats_text = self.font_small.render(
            f"Hits: {self.hits}  Misses: {self.misses}  Max Combo: {self.max_combo}",
            True,
            COLORS["white"]
        )
        self.screen.blit(stats_text, (10, 90))
        
        # Draw difficulty
        diff_text = self.font_small.render(
            f"Difficulty: {self.difficulty.capitalize()}",
            True,
            COLORS["cyan"]
        )
        self.screen.blit(diff_text, (10, 120))
        
        # Draw FPS
        fps_text = self.font_small.render(f"FPS: {self.current_fps:.1f}", True, COLORS["white"])
        self.screen.blit(fps_text, (self.screen_width - fps_text.get_width() - 10, 10))
        
        # Draw instructions overlay if enabled
        if self.show_instructions:
            self._draw_instructions()
            if time.time() > self.instructions_timer:
                self.show_instructions = False
        
        # Draw pause screen if paused
        if self.paused:
            self._draw_pause_screen()

    def _draw_instructions(self):
        """Draw game instructions overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("Cytus-style Rhythm Game", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Move your hands to hit the notes when their circle shrinks to minimum",
            "Each note has a judgment circle that shrinks over time",
            "When the circle is at its smallest, move your hand to the note",
            "Both Left and Right hands can be used to hit any note",
            "",
            f"Difficulty: {self.difficulty.capitalize()}",
            "",
            "Press ESC to pause or quit",
            "Press Space to toggle instructions"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_medium.render(line, True, COLORS["white"])
            rect = text.get_rect(center=(self.screen_width // 2, 180 + i * 40))
            self.screen.blit(text, rect)
    
    def _draw_pause_screen(self):
        """Draw pause screen overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("PAUSED", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Press P or ESC to resume",
            "Press Q to quit"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_medium.render(line, True, COLORS["white"])
            rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 20 + i * 40))
            self.screen.blit(text, rect)
    
    def spawn_note(self):
        """Randomly spawn a new note at a random position on screen."""
        # Generate random position within screen boundaries (with padding)
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
            
    def run(self):
        """Run the game loop."""
        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            if self.paused:
                                self.paused = False
                            else:
                                self.paused = True
                        elif event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_q and self.paused:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self.show_instructions = not self.show_instructions
                            if self.show_instructions:
                                self.instructions_timer = time.time() + 5.0
                
                # Capture camera frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Skip game logic if paused
                if not self.paused:
                    # Process frame
                    pose_data = self.framework.process_frame(frame)
                    
                    # Spawn new notes
                    if random.random() < self.spawn_rate:
                        self.spawn_note()
                    
                    # Check for hits
                    self.check_hits(pose_data)
                    
                    # 更新所有note状态，并移除需要移除的note
                    # 创建一个要移除的note列表
                    notes_to_remove = []
                    for note in self.notes:
                        # 如果note已经完全淡出或者判定已经完成，可能需要移除
                        if note.update():
                            notes_to_remove.append(note)
                    
                    # 移除已经完全淡出的note
                    for note in notes_to_remove:
                        if note in self.notes:
                            self.notes.remove(note)
                
                # Draw game
                self.draw_game(camera_frame=frame)
                
                # Update display
                pygame.display.flip()
                
                # Track FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_update >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_update)
                    self.fps_history.append(self.current_fps)
                    self.frame_count = 0
                    self.last_fps_update = now
                
                # Cap FPS
                self.clock.tick(60)
                
        except Exception as e:
            print(f"Error in game loop: {str(e)}")
        finally:
            # Cleanup
            pygame.quit()
            self.cap.release()
            cv2.destroyAllWindows()
            
    def quit(self):
        """Quit the game."""
        self.running = False