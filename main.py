"""
Main entry point for the 'All You Need Is Camera' project.
"""
import argparse
import sys

from aync_camera import __version__
from aync_camera.games.rhythm import RhythmGame
from aync_camera.config.rhythm_config import MUSIC_SHEETS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="All You Need Is Camera - Interactive Framework using YOLO11x-Pose"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version information and exit"
    )
    parser.add_argument(
        "--game",
        choices=["fruit_ninja", "dance", "rhythm"],
        default="rhythm",
        help="Choose which game to run (default: rhythm)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device ID to use (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x-pose.pt",
        help="Path to YOLOv8-Pose model (default: yolo11x-pose.pt)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "normal", "hard"],
        default="easy",
        help="Game difficulty level",
    )
    parser.add_argument(
        "--music",
        choices=list(MUSIC_SHEETS.keys()),
        default="earthquake",
        help="Music to use for rhythm game",
    )
    parser.add_argument(
        "--music-sheet",
        type=str,
        help="Custom music sheet JSON file path (overrides --music and --difficulty)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Game window width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Game window height (default: 720)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_args()
    
    if args.version:
        print(f"All You Need Is Camera version {__version__}")
        return 0
    
    print(f"Starting '{args.game}' game with {args.difficulty} difficulty...")
    
    if args.game == "rhythm":
        game = RhythmGame(
            camera_id=args.camera, 
            model_path=args.model,
            difficulty=args.difficulty,
            music=args.music,
            screen_width=args.width,
            screen_height=args.height,
            music_sheet_path=args.music_sheet
        )
        game.run()
    elif args.game == "fruit_ninja":
        print("Fruit Ninja game not implemented yet")
        return 1
    elif args.game == "dance":
        print("Dance Move Comparison game not implemented yet")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
