# All You Need Is Camera

一个基于YOLOv8-Pose的摄像头交互框架，让用户通过普通摄像头与游戏进行交互。

## 项目简介

"All You Need Is Camera"是一个利用计算机视觉技术，特别是人体姿态估计，让用户能够通过摄像头与游戏交互的框架。本项目使用YOLOv8-Pose模型实时检测用户的姿态，并将姿态数据转换为游戏控制信号。

目前已实现的游戏示例：
- **Falling-Note Rhythm Game**：类似跳舞毯或音乐节奏游戏，玩家通过移动手脚来击中屏幕上下落的音符。

## 安装

### 前提条件

- Python 3.10+
- 支持的操作系统：Windows、macOS、Linux
- 摄像头（笔记本内置摄像头或USB外接摄像头）
- 推荐使用GPU进行加速（不是必需的，但能提高性能）

### 安装步骤

1. 克隆仓库：
   ```
   git clone <repository_url>
   cd project
   ```

2. 使用uv安装依赖：
   ```
   uv pip install -e .
   ```

## 使用方法

### 启动节奏游戏

基本启动（使用默认设置）：
```
python main.py
```

指定难度级别：
```
python main.py --difficulty easy
```
可选的难度级别：easy（简单）、normal（普通）、hard（困难）。

指定摄像头ID（如果有多个摄像头）：
```
python main.py --camera 1
```

指定自定义YOLOv8-Pose模型：
```
python main.py --model path/to/your/model.pt
```

调整窗口大小：
```
python main.py --width 1024 --height 768
```

### 测试框架

如果想要测试PoseFramework的基本功能而不启动游戏，可以运行：
```
python test_framework.py
```

## 游戏控制

### Falling-Note Rhythm Game

游戏中有四个轨道，分别由不同的身体部位控制：
- 左脚：控制第1轨道
- 左手：控制第2轨道
- 右手：控制第3轨道
- 右脚：控制第4轨道

当音符落到屏幕底部的绿色区域时，将相应的身体部位移动到该区域以击中音符。连续击中音符将增加连击数，提高得分。

游戏按键控制：
- `ESC`：暂停/恢复游戏
- `P`：暂停/恢复游戏
- `Q`（在暂停时）：退出游戏
- `空格`：显示/隐藏游戏说明

## 项目结构

```
project/
├── aync_camera/             # 主要包目录
│   ├── core/                # 核心框架组件
│   │   └── pose_framework.py  # PoseFramework类实现
│   ├── games/               # 游戏实现
│   │   ├── game_config.py   # 游戏配置
│   │   └── rhythm_game.py   # 节奏游戏实现
│   └── utils/               # 工具函数
├── main.py                  # 主程序入口
├── test_framework.py        # 框架测试脚本
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 技术细节

- **姿态检测**：使用YOLOv8-Pose模型检测人体姿态的17个关键点，基于COCO数据集格式。
- **游戏引擎**：使用Pygame进行游戏开发。
- **视觉处理**：使用OpenCV处理摄像头输入和图像处理。

## 未来计划

- 实现更多游戏，如：
  - Fruit Ninja Clone：使用手部轨迹切水果
  - Dance Move Comparison：与标准舞蹈动作进行比较和评分
- 添加多人游戏支持
- 优化性能，提高检测准确性和响应速度
- 添加音效和背景音乐

## 贡献

欢迎贡献代码、报告问题或提出改进建议。

## 许可证

[待定]

## 致谢

- 本项目使用了[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)模型进行姿态检测
- 项目灵感来源于类似体感游戏和交互式健身应用