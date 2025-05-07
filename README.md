# All You Need Is Camera

An interactive framework based on YOLOv11-Pose that allows users to interact with games using just a standard camera-no special equipment required.

## Project Overview

**'All You Need Is Camera'** is a framework that utilizes computer vision technology to enable users to interact with games through a camera. This project uses the **YOLOv11-Pose** model to detect the user's pose in real-time and converts pose data into game control signals.

Currently implemented:
- **Rhythm Game**: Players move their hands and feet to hit notes on the screen, similar to dance mat games.

## Features

- **Game Generation**: Dynamic game note generation based on upload video.
- **User Calibration**: Calibration of user position and camera angle using **ViT**.
- **Pose Detection**: Uses **YOLOv11-Pose** to detect 17 key points and translate them into game controls.
- **Object Interaction**: Users can use objects like a **fan** or **basketball** to interact with notes.
- **Types of Notes**: Hit notes, swap notes, hold notes, and turn-around notes.

## Installation

### Prerequisites
- Python 3.10+
- Camera (laptop built-in or external)

### Installation Steps
1. Clone the repository:
   ```
   git clone <repository_url>
   cd project
   ```

2. Install dependencies:
   ```
   uv sync
   ```

## Usage

### Creating Music Sheets
1. Run the generator script:
   ```
   python writemusicsheet.py
   ```
2. Follow the prompts to set up video path and parameters.
3. Output files will be stored in `musicsheet/<video_name>/` directory.

### Launch the Game
```
python main.py
```

### Test the Framework
```
python test_framework.py
```

## Game Controls

- `ESC` or `P`: Pause/Resume game
- `Q` (while paused): Quit game
- `Space`: Show/Hide instructions

## Project Structure
```
project/
├── aync_camera/              # Main package directory
│   ├── common/               # Common components
│   ├── config/               # Configuration modules
│   ├── core/                 # Core framework components
│   ├── games/                # Game implementations
│   ├── initialization/       # User setup
│   └── ui/                   # User interface components
├── main.py                   # Main entry point
├── writemusicsheet.py        # Music sheet generator
├── test_framework.py         # Framework test
├── musicsheet/               # Generated music sheets
└── example_video/            # Example videos
```

## Technical Details

- **Pose Detection**: YOLOv11-Pose model to detect 17 key points
- **Game Engine**: Pygame for game development
- **Visual Processing**: OpenCV for camera input and processing
- **ViT Calibration**: Vision Transformer for user distance and angle detection
- **Beat Detection**: Librosa for audio beat detection from videos
- **Pose Synchronization**: Extracts dancer poses at each beat point

## Architecture Design

The project follows a modular architecture with:
1. **Core Framework**: Pose detection capabilities
2. **Game Base**: Common interfaces for all games
3. **Game Logic**: Game mechanics and state management
4. **UI Rendering**: Visual presentation components
5. **Configuration**: Settings and parameters

## Future Plans

- Additional games: Fruit Ninja, Dance Move Comparison, Fitness games
- Multiplayer support
- Performance optimization
- Enhanced audio features

## Contributing

We welcome contributions in the form of code, issue reports, or suggestions for improvements.

## License

This project is licensed under the **MIT License**.

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for pose detection
- Project inspiration from motion-sensing games and interactive fitness applications