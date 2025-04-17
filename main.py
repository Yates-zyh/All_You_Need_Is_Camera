"""
Main entry point for the 'All You Need Is Camera' project.
"""
import argparse
import sys

from aync_camera import __version__
from aync_camera.games import FallingNoteGame


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="All You Need Is Camera - Interactive Framework using YOLOv8-Pose"
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
        default="yolov8n-pose.pt",
        help="Path to YOLOv8-Pose model (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "normal", "hard"],
        default="normal",
        help="Game difficulty level (default: normal)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Game window width (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Game window height (default: 600)",
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
        game = FallingNoteGame(
            camera_id=args.camera, 
            model_path=args.model,
            difficulty=args.difficulty,
            screen_width=args.width,
            screen_height=args.height
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
