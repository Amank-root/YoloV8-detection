# Bar Detection and Conveyor Movement Monitoring

This project uses YOLOv8 to detect metal bars on a conveyor belt and monitor if the conveyor has stopped moving. It provides visual alerts when the conveyor is halted.

## Output

https://github.com/user-attachments/assets/07570ad0-e790-46f1-be33-dae3d2ad3ffa
---

## Features

- Real-time metal bar detection using YOLOv8
- Conveyor belt movement monitoring
- Visual alerts for conveyor stoppages
- Works with both video files and live camera feeds

## Project Structure

```
├── model/
│   └── best.pt        # YOLOv8 trained model
├── assets/
│   └── conv.mp4       # Sample video for testing
├── main.py            # Main application file
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

## Setup

### Virtual Environment Setup

1. Create a virtual environment:

```bash
# Windows
python -m venv venv

# Linux/macOS
python3 -m venv venv
```

2. Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python main.py
```

2. Configuration Options:

The following settings can be modified in main.py:

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to your YOLOv8 model |
| `video_path` | Path to video file or camera index |
| `output_path` | Path to save the output video |
| `stop_threshold` | Number of frames to determine if conveyor is stopped |
| `movement_tolerance` | Pixel tolerance for movement detection |

3. For live camera detection:
   - Uncomment the line `# video_path = 0` in the configuration section

## How It Works

1. **Object Detection**: The system uses a pre-trained YOLOv8 model to detect metal bars in each frame.

2. **Movement Tracking**: For each detected bar, the system tracks its centroid position across frames.

3. **Stop Detection**: The system calculates the average movement of all tracked objects between consecutive frames:
   - If movement falls below the `movement_tolerance` threshold, a counter is incremented
   - When the counter exceeds the `stop_threshold`, the conveyor is considered stopped

4. **Visual Alerts**: 
   - Detected bars are highlighted with yellow rectangles
   - "OK" (green) indicates the conveyor is running normally
   - "Conveyor Halted" (red) warning appears when the system detects a stoppage
