# All You Need Is Camera

An interactive framework based on YOLOv8-Pose that allows users to interact with games using a standard camera.

## Project Overview

"All You Need Is Camera" is a framework that utilizes computer vision technology, specifically human pose estimation, to enable users to interact with games through a camera. This project uses the YOLOv8-Pose model to detect the user's pose in real-time and converts pose data into game control signals.

Currently implemented game examples:
- **Falling-Note Rhythm Game**: Similar to dance mats or music rhythm games, players move their hands and feet to hit notes falling on the screen.

## Installation

### Prerequisites

- Python 3.10+
- Supported operating systems: Windows, macOS, Linux
- Camera (laptop built-in camera or USB external camera)
- GPU acceleration recommended (not required, but improves performance)

### Installation Steps

1. Clone the repository:
   ```
   git clone <repository_url>
   cd project
   ```

2. Install dependencies using uv:
   ```
   uv sync
   ```

## Usage

### Launch the Rhythm Game

Basic launch (with default settings):
```
python main.py
```

Specify difficulty level:
```
python main.py --difficulty easy
```
Available difficulty levels: easy, normal, hard.

Specify camera ID (if you have multiple cameras):
```
python main.py --camera 1
```

Specify a custom YOLOv8-Pose model:
```
python main.py --model path/to/your/model.pt
```

Adjust window size:
```
python main.py --width 1024 --height 768
```

### Test the Framework

If you want to test the basic functionality of PoseFramework without launching a game, run:
```
python test_framework.py
```

## Game Controls

### Falling-Note Rhythm Game

In the game, there are four lanes, each controlled by different body parts:
- Left hand: Controls lanes 0 and 1
- Right hand: Controls lanes 2 and 3

When notes reach the judgment area, move the corresponding body part to that area to hit the note. Consecutive hits will increase your combo and score.

Game key controls:
- `ESC`: Pause/Resume game
- `P`: Pause/Resume game
- `Q` (while paused): Quit game
- `Space`: Show/Hide game instructions

## Project Structure

```
project/
├── aync_camera/              # Main package directory
│   ├── common/               # Common components and base classes
│   │   └── game_base.py      # Base class for all games
│   ├── config/               # Configuration modules
│   │   ├── game_settings.py  # Common game settings
│   │   └── rhythm_config.py  # Rhythm game specific settings
│   ├── core/                 # Core framework components
│   │   └── pose_framework.py # PoseFramework implementation
│   ├── games/                # Game implementations
│   │   └── rhythm/           # Rhythm game modules
│   │       ├── __init__.py   # Main rhythm game class
│   │       ├── game_logic.py # Game logic implementation
│   │       ├── note.py       # Note class implementation
│   │       └── music_sheet_loader.py # Music sheet loading functionality
│   ├── initialization/       # User and system initialization
│   └── ui/                   # User interface components
│       └── rhythm/           # Rhythm game UI
│           └── renderer.py   # UI rendering for rhythm game
├── main.py                   # Main program entry
├── test_framework.py         # Framework test script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Technical Details

- **Pose Detection**: Uses YOLOv8-Pose model to detect 17 key points of human pose, based on COCO dataset format.
- **Game Engine**: Uses Pygame for game development.
- **Visual Processing**: Uses OpenCV for camera input and image processing.

## Architecture Design

The project follows a modular architecture with clear separation of concerns:

1. **Core Framework**: Provides pose detection and tracking capabilities
2. **Game Base**: Defines common interfaces and functionality for all games
3. **Game Logic**: Handles game mechanics, rules, and state management
4. **UI Rendering**: Manages visual presentation and user interface elements
5. **Configuration**: Stores game settings and parameters separately from code

This architecture makes the code more maintainable and extensible, allowing for easy addition of new games and features.

## Future Plans

- Implement more games, such as:
  - Fruit Ninja Clone: Use hand trajectories to slice fruits
  - Dance Move Comparison: Compare and score against standard dance moves
- Add multiplayer support
- Optimize performance, improve detection accuracy and response speed
- Add sound effects and background music

## Contributing

Contributions of code, issue reports, or improvement suggestions are welcome.

## License

[TBD]

## Acknowledgments

- This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for pose detection
- Project inspiration comes from similar motion-sensing games and interactive fitness applications