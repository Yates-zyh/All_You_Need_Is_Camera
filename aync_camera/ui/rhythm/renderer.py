"""
Rhythm game UI rendering module, responsible for drawing the game interface.
"""
import time
import cv2
import pygame

from aync_camera.config.game_settings import COLORS

# Game state constants
START_SCREEN = "start_screen"
SONG_SELECTION = "song_selection"
COUNTDOWN = "countdown"
PLAYING = "playing"
PAUSED = "paused"
GAME_OVER = "game_over"

class RhythmGameRenderer:
    """Rhythm game UI renderer, handles drawing of all UI elements."""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the UI renderer.
        
        Args:
            screen_width: Game window width
            screen_height: Game window height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Instruction overlay flag
        self.show_instructions = True
        self.instructions_timer = time.time() + 5.0  # Show for 5 seconds
        
        # Mirror mode flag (enabled by default)
        self.mirror_mode = True
        
        # Difficulty selection buttons
        button_width = 300
        button_height = 60
        button_spacing = 20
        self.difficulty_buttons = {
            "easy": pygame.Rect(self.screen_width // 2 - button_width // 2, 
                              self.screen_height // 2 - button_height - button_spacing, 
                              button_width, button_height),
            "normal": pygame.Rect(self.screen_width // 2 - button_width // 2, 
                               self.screen_height // 2, 
                               button_width, button_height),
            "hard": pygame.Rect(self.screen_width // 2 - button_width // 2, 
                              self.screen_height // 2 + button_height + button_spacing, 
                              button_width, button_height)
        }
        
        # Song selection buttons
        self.song_buttons = {
            "earthquake": pygame.Rect(self.screen_width // 2 - button_width // 2, 
                                   self.screen_height // 2 - button_height - button_spacing, 
                                   button_width, button_height),
            "maria": pygame.Rect(self.screen_width // 2 - button_width // 2, 
                              self.screen_height // 2, 
                              button_width, button_height)
        }
    
    def render_game(self, screen, game_state, camera_frame=None):
        """
        Render the complete game screen.
        
        Args:
            screen: Pygame screen object
            game_state: Game state dictionary
            camera_frame: Optional camera frame for background display
        """
        # Clear the screen
        screen.fill(COLORS["black"])
        
        # If a camera frame is provided, use it as the background
        if camera_frame is not None:
            # Get the appropriate frame based on mirror mode
            if 'original_frame' in camera_frame and not self.mirror_mode:
                # Use original frame when mirror mode is off
                display_frame = camera_frame['original_frame']
            elif 'mirrored_frame' in camera_frame and self.mirror_mode:
                # Use mirrored frame when mirror mode is on
                display_frame = camera_frame['mirrored_frame']
            else:
                # Fallback to direct frame if neither is available
                display_frame = camera_frame
                
            # Resize to fit the screen
            display_frame = cv2.resize(display_frame, (self.screen_width, self.screen_height))
            # Convert to RGB for pygame
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
            # Add transparency
            surface.set_alpha(160)  # Slightly less opaque for better visibility of poses
            # Draw
            screen.blit(surface, (0, 0))
        
        # Draw notes
        for note in game_state['notes']:
            note.draw(screen)
        
        # Draw hit feedback animation
        self.render_hit_feedback(screen, game_state['hit_feedback'], game_state['hit_feedback_duration'])
        
        # Draw score and combo
        self.render_score(screen, game_state['score'], game_state['combo'])
        
        # Draw statistics
        self.render_stats(screen, game_state['hits'], game_state['misses'], game_state['max_combo'], game_state['difficulty'])
        
        # Draw FPS (if provided)
        if 'fps' in game_state:
            self.render_fps(screen, game_state['fps'])
    
    def render_score(self, screen, score, combo):
        """
        Render score and combo.
        
        Args:
            screen: Pygame screen object
            score: Current score
            combo: Current combo
        """
        # Draw score
        score_text = self.font_medium.render(f"Score: {score}", True, COLORS["white"])
        screen.blit(score_text, (10, 10))
        
        # Draw combo
        combo_text = self.font_medium.render(f"Combo: {combo}", True, COLORS["yellow"])
        screen.blit(combo_text, (10, 50))
    
    def render_stats(self, screen, hits, misses, max_combo, difficulty):
        """
        Render game statistics.
        
        Args:
            screen: Pygame screen object
            hits: Successful hits count
            misses: Misses count
            max_combo: Maximum combo
            difficulty: Game difficulty
        """
        # Draw statistics
        stats_text = self.font_small.render(
            f"Hits: {hits}  Misses: {misses}  Max Combo: {max_combo}",
            True,
            COLORS["white"]
        )
        screen.blit(stats_text, (10, 90))
        
        # Draw difficulty
        diff_text = self.font_small.render(
            f"Difficulty: {difficulty.capitalize()}",
            True,
            COLORS["cyan"]
        )
        screen.blit(diff_text, (10, 120))
    
    def render_fps(self, screen, fps):
        """
        Render FPS information.
        
        Args:
            screen: Pygame screen object
            fps: Current FPS value
        """
        fps_text = self.font_small.render(f"FPS: {fps:.1f}", True, COLORS["white"])
        screen.blit(fps_text, (self.screen_width - fps_text.get_width() - 10, 10))
    
    def render_hit_feedback(self, screen, hit_feedback, hit_feedback_duration):
        """
        Render hit feedback animation.
        
        Args:
            screen: Pygame screen object
            hit_feedback: Hit feedback data list
            hit_feedback_duration: Hit feedback duration
        """
        current_time = time.time()
        
        for hit_x, hit_y, hit_time, hit_type in hit_feedback:
            # Calculate animation progress (0.0 to 1.0)
            elapsed = current_time - hit_time
            if elapsed > hit_feedback_duration:
                continue
                
            progress = elapsed / hit_feedback_duration
            size = int(50 * (1.0 - progress))  # Start large, shrink as animation progresses
            alpha = int(255 * (1.0 - progress))  # Fade out
            
            # Draw hit effect
            color = COLORS["green"] if hit_type == "hit" else COLORS["red"]
            hit_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(hit_surface, (*color, alpha), (size, size), size)
            screen.blit(hit_surface, (hit_x - size, hit_y - size))
            
            # Draw "HIT!" text (float upwards)
            text = self.font_medium.render("HIT!", True, color)
            text_surface = pygame.Surface(text.get_size(), pygame.SRCALPHA)
            text_surface.fill((0, 0, 0, 0))  # Transparent
            text_surface.blit(text, (0, 0))
            text_surface.set_alpha(alpha)
            offset_y = int(30 * progress)  # Text floats upwards
            screen.blit(text_surface, (hit_x - text.get_width() // 2, hit_y - text.get_height() - offset_y))
    
    def render_instructions(self, screen):
        """
        Render game instructions overlay.
        
        Args:
            screen: Pygame screen object
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Black with transparency
        screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("Rhythm Game", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        screen.blit(title, title_rect)
        
        # Instructions
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
        Render pause screen overlay.
        
        Args:
            screen: Pygame screen object
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Black with transparency
        screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("PAUSED", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(title, title_rect)
        
        # Instructions
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
        Check whether instructions should be displayed.
        
        Returns:
            bool: True if instructions should be displayed, False otherwise
        """
        if self.show_instructions and time.time() > self.instructions_timer:
            self.show_instructions = False
        
        return self.show_instructions
    
    def toggle_instructions(self):
        """Toggle the instructions display state."""
        self.show_instructions = not self.show_instructions
        if self.show_instructions:
            self.instructions_timer = time.time() + 5.0
    
    def render_start_screen(self, screen, selected_song=None):
        """
        Render the game start screen.
        
        Args:
            screen: Pygame screen object
            selected_song: Currently selected song name, displayed in the difficulty selection screen
        """
        # Clear the screen and draw the background
        screen.fill(COLORS["black"])
        
        # Draw game title
        title = self.font_large.render("Rhythm Pose Game", True, COLORS["cyan"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        screen.blit(title, title_rect)
        
        # Draw song name
        if selected_song:
            song_text = self.font_medium.render(f"Song: {selected_song}", True, COLORS["white"])
            song_rect = song_text.get_rect(center=(self.screen_width // 2, 170))
            screen.blit(song_text, song_rect)
        
        # Draw prompt text
        subtitle = self.font_medium.render("Select Difficulty", True, COLORS["white"])
        subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 100))
        screen.blit(subtitle, subtitle_rect)
        
        # Draw difficulty selection buttons
        difficulty_texts = {
            "easy": "Easy",
            "normal": "Normal",
            "hard": "Hard"
        }
        
        for difficulty, rect in self.difficulty_buttons.items():
            # Draw button background
            button_color = COLORS["green"] if difficulty == "easy" else (
                COLORS["yellow"] if difficulty == "normal" else COLORS["red"])
            pygame.draw.rect(screen, button_color, rect, border_radius=10)
            pygame.draw.rect(screen, COLORS["white"], rect, 2, border_radius=10)  # Border
            
            # Draw button text
            button_text = self.font_medium.render(difficulty_texts[difficulty], True, COLORS["black"])
            text_rect = button_text.get_rect(center=rect.center)
            screen.blit(button_text, text_rect)
    
    def render_song_selection(self, screen):
        """
        Render the song selection screen.
        
        Args:
            screen: Pygame screen object
        """
        # Clear the screen and draw the background
        screen.fill(COLORS["black"])
        
        # Draw title
        title = self.font_large.render("Select Song", True, COLORS["cyan"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        screen.blit(title, title_rect)
        
        # Draw song buttons
        song_texts = {
            "earthquake": "Earthquake",
            "maria": "Maria"
        }
        
        for song, rect in self.song_buttons.items():
            # Draw button background
            pygame.draw.rect(screen, COLORS["blue"], rect, border_radius=10)
            pygame.draw.rect(screen, COLORS["white"], rect, 2, border_radius=10)  # Border
            
            # Draw button text
            button_text = self.font_medium.render(song_texts[song], True, COLORS["white"])
            text_rect = button_text.get_rect(center=rect.center)
            screen.blit(button_text, text_rect)
    
    def render_countdown(self, screen, countdown_time, camera_frame=None):
        """
        Render the countdown screen.
        
        Args:
            screen: Pygame screen object
            countdown_time: Remaining countdown time (seconds)
            camera_frame: Optional camera frame for background display
        """
        # Clear the screen
        screen.fill(COLORS["black"])
        
        # If a camera frame is provided, use it as the background
        if camera_frame is not None:
            # Apply horizontal flip to create mirror effect
            # Always use mirrored frame for countdown to help with user orientation
            if isinstance(camera_frame, dict) and 'original_frame' in camera_frame:
                display_frame = cv2.flip(camera_frame['original_frame'], 1)  # 1 means horizontal flip
            else:
                # If we got a direct frame, flip it
                display_frame = cv2.flip(camera_frame, 1)
                
            # Resize to fit the screen
            display_frame = cv2.resize(display_frame, (self.screen_width, self.screen_height))
            # Convert to RGB for pygame
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
            # Add transparency
            surface.set_alpha(160)  # Slightly less opaque for better visibility of poses
            # Draw
            screen.blit(surface, (0, 0))
        
        # Determine countdown number
        countdown_value = max(0, min(3, int(countdown_time) + 1))
        if countdown_value == 0:
            display_text = "Start!"
        else:
            display_text = str(countdown_value)
        
        # Draw countdown number (large font)
        text_color = COLORS["green"] if countdown_value == 0 else COLORS["white"]
        countdown_font = pygame.font.Font(None, 200)  # Larger font
        countdown_text = countdown_font.render(display_text, True, text_color)
        text_rect = countdown_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        
        # Add background circle to make the number more prominent
        circle_radius = max(countdown_text.get_width(), countdown_text.get_height()) // 2 + 20
        circle_surface = pygame.Surface((circle_radius * 2, circle_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 0, 180), (circle_radius, circle_radius), circle_radius)
        screen.blit(circle_surface, (text_rect.centerx - circle_radius, text_rect.centery - circle_radius))
        
        screen.blit(countdown_text, text_rect)
    
    def toggle_mirror_mode(self):
        """Toggle mirror mode."""
        self.mirror_mode = not self.mirror_mode

    def render_game_over_screen(self, screen, final_score, hits, misses, max_combo):
        """
        Render the game over screen.
        
        Args:
            screen: Pygame screen object
            final_score: Final game score
            hits: Number of successful hits
            misses: Number of misses
            max_combo: Maximum combo achieved
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Black with transparency
        screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("GAME OVER", True, COLORS["white"])
        title_rect = title.get_rect(center=(self.screen_width // 2, 120))
        screen.blit(title, title_rect)
        
        # Score and statistics
        score_text = self.font_large.render(f"Score: {final_score}", True, COLORS["yellow"])
        score_rect = score_text.get_rect(center=(self.screen_width // 2, 200))
        screen.blit(score_text, score_rect)
        
        stats_text = self.font_medium.render(
            f"Hits: {hits}   Misses: {misses}   Max Combo: {max_combo}", 
            True, 
            COLORS["white"]
        )
        stats_rect = stats_text.get_rect(center=(self.screen_width // 2, 260))
        screen.blit(stats_text, stats_rect)
        
        # Draw restart button
        restart_button = pygame.Rect(
            self.screen_width // 2 - 150, 
            self.screen_height // 2 + 50, 
            300, 
            60
        )
        pygame.draw.rect(screen, COLORS["green"], restart_button, border_radius=10)
        pygame.draw.rect(screen, COLORS["white"], restart_button, 2, border_radius=10)  # Border
        
        restart_text = self.font_medium.render("Select New Song", True, COLORS["black"])
        restart_rect = restart_text.get_rect(center=restart_button.center)
        screen.blit(restart_text, restart_rect)
        
        # Draw exit button
        exit_button = pygame.Rect(
            self.screen_width // 2 - 150, 
            self.screen_height // 2 + 130, 
            300, 
            60
        )
        pygame.draw.rect(screen, COLORS["red"], exit_button, border_radius=10)
        pygame.draw.rect(screen, COLORS["white"], exit_button, 2, border_radius=10)  # Border
        
        exit_text = self.font_medium.render("Exit Game", True, COLORS["white"])
        exit_rect = exit_text.get_rect(center=exit_button.center)
        screen.blit(exit_text, exit_rect)
        
        # Store button positions for click handling
        self.restart_button = restart_button
        self.exit_button = exit_button