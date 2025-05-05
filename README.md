# All You Need Is Camera

An interactive framework based on YOLOv11-Pose that allows users to interact with games using just a standard camera-no special equipment required.

## Project Overview

**'All You Need Is Camera'** is a framework that utilizes computer vision technology, specifically human pose estimation, to enable users to interact with games through a camera. This project uses the **YOLOv11-Pose** model to detect the user's pose in real-time and converts pose data into game control signals. The best part? No special equipment is required-just a standard camera.

Currently implemented game examples:

- **Cytus-like Rhythm Game**: Similar to dance mats or music rhythm games, players move their hands and feet to hit notes on the screen. In this game, notes fall along the screen, and players must position their hands or feet in the appropriate positions to 'hit' the notes as they appear in the judgment area.

## Features

- **Game Generation**: Dynamic game note generation based on upload video.
- **User Calibration**: Calibration of user position and camera angle using **ViT** to minimize the impact of the camera's position and angle.
- **Pose Detection**: The system uses **YOLOv11-Pose** to detect 17 key points of human pose and translates them into interactive game controls.
- **Object Interaction**: In addition to body pose interaction, users can use objects like a **fan** or **basketball** to interact with the notes and further personalize the dance map.
- **Types of Notes**: There are different types of notes in the game such as hit notes, swap notes, hold notes, and turn-around notes.

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

Specify a custom YOLOv11-Pose model:

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

### Cytus-like Rhythm Game

Types of Notes: The game features different types of notes that the user needs to interact with:

-    Hit Notes: Regular notes that need to be hit.
-    Swap Notes: Notes that require a hand or foot to move to another position.
-    Hold Notes: Notes that require the user to hold their position for a duration.
-    Turn-Around Notes: Notes that require the user to turn around to face the camera.

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

- **Pose Detection**: Uses YOLOv11-Pose model to detect 17 key points of human pose, based on COCO dataset format.
- **Game Engine**: Uses Pygame for game development.
- **Visual Processing**: Uses OpenCV for camera input and image processing. and 关键帧和节奏获取
- **ViT Calibration**: Uses ViT (Vision Transformer) to detect user distance and angles to ensure proper calibration for optimal gameplay experience.
- **DNN Scoring**: Deep Neural Networks (DNN) are used to score the accuracy and fluidity of user movements.

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
  - fitness game
- Add multiplayer support
- Optimize performance, improve detection accuracy and response speed
- Add sound effects and background music

## Contributing

We welcome contributions in the form of code, issue reports, or suggestions for improvements. Feel free to fork the repository, submit pull requests, or open issues for any bugs or feature requests.

## License

This project is licensed under the **MIT License**.

## Acknowledgments

- This project uses the [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) model for pose detection
- Project inspiration comes from similar motion-sensing games and interactive fitness applications