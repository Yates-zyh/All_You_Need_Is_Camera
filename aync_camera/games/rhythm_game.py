"""
Falling-Note Rhythm Game implementation using the PoseFramework.
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
    """Represents a falling note in the rhythm game."""
    
    def __init__(self, lane: int, x: int, y: int, width: int, height: int, speed: float = 5.0):
        """
        Initialize a new note.
        
        Args:
            lane: Lane index (0-3)
            x: X-coordinate
            y: Y-coordinate
            width: Note width
            height: Note height
            speed: Falling speed in pixels per frame
        """
        self.lane = lane
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.rect = pygame.Rect(x, y, width, height)
        self.hit = False
        self.missed = False
        
    def update(self, screen_height: int):
        """
        Update note position.
        
        Args:
            screen_height: Height of the screen
            
        Returns:
            True if note should be removed (offscreen), False otherwise
        """
        self.y += self.speed
        self.rect.y = self.y
        
        # Check if note is offscreen
        if self.y > screen_height:
            self.missed = True
            return True
            
        return False
    
    def draw(self, screen):
        """
        Draw the note.
        
        Args:
            screen: Pygame screen to draw on
        """
        color = COLORS["blue"]  # Blue for regular notes
        if self.hit:
            color = COLORS["green"]  # Green for hit notes
        elif self.missed:
            color = COLORS["red"]  # Red for missed notes
            
        pygame.draw.rect(screen, color, self.rect)
        # Add a border
        pygame.draw.rect(screen, COLORS["white"], self.rect, 2)


class FallingNoteGame:
    """Falling-Note Rhythm Game using body pose as controls."""
    
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
        self.num_lanes = 4
        self.lane_width = self.screen_width // self.num_lanes
        self.hit_zone_height = 100  # Increased hit zone height for easier detection
        self.note_width = self.lane_width - 20  # 10px padding on each side
        self.note_height = 30
        self.note_speed = self.difficulty_settings["note_speed"]
        self.spawn_rate = self.difficulty_settings["spawn_rate"]
        self.confidence_threshold = self.difficulty_settings["confidence_threshold"]
        
        # Game state
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.game_over = False
        self.paused = False
        
        # Create hit zones (rectangles at the bottom of each lane)
        self.hit_zones = [
            pygame.Rect(
                i * self.lane_width, 
                self.screen_height - self.hit_zone_height, 
                self.lane_width, 
                self.hit_zone_height
            ) for i in range(self.num_lanes)
        ]
        
        # Vertical detection zones for each lane (full height columns)
        self.vertical_lanes = [
            pygame.Rect(
                i * self.lane_width, 
                0, 
                self.lane_width, 
                self.screen_height
            ) for i in range(self.num_lanes)
        ]
        
        # Active notes
        self.notes = []
        
        # Palm hit feedback (visual indicator for successful hits)
        self.hit_feedback = []  # List of (x, y, time, lane) tuples for hit animations
        self.hit_feedback_duration = 0.5  # Duration in seconds to show hit feedback
        
        # Setup screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Falling-Note Rhythm Game - {difficulty.capitalize()} Mode")
        self.clock = pygame.time.Clock()
        
        # Initialize pose framework
        self.framework = PoseFramework(
            model_path=model_path, 
            confidence_threshold=self.confidence_threshold
        )
        self.cap = self.framework.setup_camera(camera_id=camera_id)
        
        # Keypoint mapping (which keypoint controls which lane)
        self.lane_keypoint_map = DEFAULT_LANE_KEYPOINT_MAP.copy()
        
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
        
        # Check each lane
        for lane in range(self.num_lanes):
            # Get the corresponding keypoint for this lane
            keypoint_idx = self.lane_keypoint_map.get(lane)
            if keypoint_idx not in palm_positions:
                continue
            
            # Get palm position
            px, py = palm_positions[keypoint_idx]
            
            # Check if palm is in the vertical lane
            if not self.vertical_lanes[lane].collidepoint(px, py):
                continue
                
            # Check for note collision in this lane
            for note in self.notes[:]:
                if note.lane == lane and note.y > self.screen_height * 0.6 and not note.hit:
                    # Only check notes that have fallen past 60% of the screen height
                    # and are in the vertical lane where the palm is detected
                    
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
                    
                    # Mark note as hit and remove
                    note.hit = True
                    self.notes.remove(note)
                    
                    # Add visual hit feedback
                    self.hit_feedback.append((px, py, time.time(), lane))
                    break

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
        
        # Draw lanes and hit zones
        for i in range(self.num_lanes):
            # Draw lane dividers
            pygame.draw.line(
                self.screen,
                COLORS["white"],
                (i * self.lane_width, 0),
                (i * self.lane_width, self.screen_height),
                2
            )
            
            # Draw vertical lane detection zones (semi-transparent)
            lane_surface = pygame.Surface((self.lane_width, self.screen_height), pygame.SRCALPHA)
            lane_color = (0, 255, 0, 40) if i < 2 else (0, 0, 255, 40)  # Green for left, Blue for right
            pygame.draw.rect(lane_surface, lane_color, pygame.Rect(0, 0, self.lane_width, self.screen_height))
            self.screen.blit(lane_surface, (i * self.lane_width, 0))
            
            # Draw hit zones (more visible)
            hit_zone_surface = pygame.Surface((self.lane_width, self.hit_zone_height), pygame.SRCALPHA)
            hit_zone_color = (0, 255, 0, 80) if i < 2 else (0, 0, 255, 80)
            pygame.draw.rect(hit_zone_surface, hit_zone_color, pygame.Rect(0, 0, self.lane_width, self.hit_zone_height))
            self.screen.blit(hit_zone_surface, (i * self.lane_width, self.screen_height - self.hit_zone_height))
            
            # Draw lane labels
            kp_idx = self.lane_keypoint_map.get(i)
            if kp_idx in KEYPOINT_LABELS:
                label = KEYPOINT_LABELS[kp_idx]
                text = self.font_small.render(label, True, COLORS["white"])
                x = i * self.lane_width + (self.lane_width - text.get_width()) // 2
                self.screen.blit(text, (x, self.screen_height - self.hit_zone_height + 5))
        
        # Draw notes
        for note in self.notes:
            note.draw(self.screen)
        
        # Draw hit feedback animations
        current_time = time.time()
        for hit_x, hit_y, hit_time, lane in self.hit_feedback[:]:
            # Calculate animation progress (0.0 to 1.0)
            elapsed = current_time - hit_time
            if elapsed > self.hit_feedback_duration:
                self.hit_feedback.remove((hit_x, hit_y, hit_time, lane))
                continue
                
            progress = elapsed / self.hit_feedback_duration
            size = int(50 * (1.0 - progress))  # Start big, shrink as animation progresses
            alpha = int(255 * (1.0 - progress))  # Fade out
            
            # Draw hit effect
            color = COLORS["green"] if lane < 2 else COLORS["cyan"]
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
        title = self.font_large.render("Falling-Note Rhythm Game", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Move your hands to hit the falling notes",
            "Left Hand controls Lanes 1 & 2",
            "Right Hand controls Lanes 3 & 4",
            "Move your palm into a lane when a note reaches the bottom",
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
        """Randomly spawn a new note."""
        if random.random() < self.spawn_rate:
            lane = random.randint(0, self.num_lanes - 1)
            x = lane * self.lane_width + (self.lane_width - self.note_width) // 2
            note = Note(
                lane=lane,
                x=x,
                y=0,
                width=self.note_width,
                height=self.note_height,
                speed=self.note_speed
            )
            self.notes.append(note)
            
    def update_notes(self):
        """Update all active notes."""
        for note in self.notes[:]:  # Iterate over a copy for safe removal
            if note.update(self.screen_height):
                # Note is offscreen or needs to be removed
                if note.missed and not note.hit:
                    # Note was missed
                    self.misses += 1
                    self.combo = 0
                self.notes.remove(note)
    
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
                    self.spawn_note()
                    
                    # Update notes
                    self.update_notes()
                    
                    # Check for hits
                    self.check_hits(pose_data)
                
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