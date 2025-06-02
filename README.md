# szpuszi-AI-object-detector

Real-time object detection system using YOLOv8 with multi-category recognition and colorful visualization.

## Features

- Real-time object detection using YOLOv8
- Support for multiple object categories:
  - People (person, face, eyes, etc.)
  - Electronics (phone, laptop, keyboard, etc.)
  - Furniture (chair, couch, bed, etc.)
  - Everyday objects (bottle, cup, book, etc.)
  - Accessories (backpack, umbrella, handbag, etc.)
  - Food items (banana, apple, sandwich, etc.)
  - Vehicles (bicycle, car, motorcycle)
  - Animals (dog, cat, bird)
- Color-coded detection boxes for different categories
- Real-time FPS counter
- Object count display
- GPU acceleration support (CUDA)
- High-resolution camera support (1280x720)

## Requirements

- Python 3.8 or newer
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/szpuszi/szpuszi-AI-object-detector.git
cd szpuszi-AI-object-detector
```

2. Install required packages:
```bash
pip install opencv-python
pip install torch torchvision
pip install ultralytics
```

3. Download YOLOv8 model:
```bash
# The model will be downloaded automatically on first run
# or you can manually download yolov8x.pt from Ultralytics
```

## Usage

1. Run the program:
```bash
python AI.py
```

2. Controls:
   - Press 'q' to quit
   - The program will automatically use GPU if available

## How It Works

1. The program initializes your webcam
2. YOLOv8 model processes each frame
3. Detected objects are highlighted with colored boxes
4. Object counts and FPS are displayed in real-time
5. Different colors are used for different object categories

## Technical Details

### Detection Categories
- Person-related: person, face, eye, nose, mouth, ear, hand, arm, leg, foot
- Electronics: cell phone, laptop, keyboard, mouse, remote, tv, monitor
- Furniture: chair, couch, bed, dining table, toilet, sink
- Objects: bottle, cup, book, clock, vase, scissors, toothbrush, hair drier
- Accessories: backpack, umbrella, handbag, tie
- Food: banana, apple, sandwich, orange, pizza
- Vehicles: bicycle, car, motorcycle
- Animals: dog, cat, bird

### Performance
- Uses YOLOv8x model for high accuracy
- Supports GPU acceleration via CUDA
- Real-time processing with FPS display
- Confidence threshold: 0.45
- IOU threshold: 0.45

## Security Features

- Safe camera handling
- Error handling for camera access
- Graceful program termination
- Resource cleanup on exit

## Contributing

Feel free to submit issues and enhancement requests!

## Credits

Created by [szpuszi](https://github.com/szpuszi)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
