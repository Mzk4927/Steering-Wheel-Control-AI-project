# Steering Wheel Control & Asphalt Controller

This project uses computer vision and pose estimation to control a racing game using your body movements via webcam. It leverages OpenCV, MediaPipe, and keyboard automation to simulate steering and nitro boost actions.

## Features
- **Steering Wheel Control (`app.py`)**: Move your hands up/down to steer left/right, and trigger nitro by raising your right hand above your nose.
- **Asphalt Controller (`test.py`)**: Lean forward to accelerate, lean left/right to steer, and raise your right hand for nitro boost.

## Requirements
- Python 3.10+
- OpenCV
- MediaPipe
- keyboard

## Setup
1. Create and activate the virtual environment:
   ```powershell
   python -m venv venv310
   .\venv310\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install opencv-python mediapipe keyboard
   ```

## Usage
- Run `app.py` for steering wheel control:
  ```powershell
  python app.py
  ```
- Run `test.py` for asphalt controller:
  ```powershell
  python test.py
  ```

## Controls
### app.py
- **Steer Left/Right**: Raise left/right hand
- **Forward**: Hands at same level
- **Nitro**: Right hand above nose
- **Quit**: Press 'q'

### test.py
- **Accelerate**: Lean forward
- **Steer**: Lean left/right
- **Nitro**: Right hand above nose
- **Quit**: Press 'q'

## Notes
- Make sure your webcam is connected.
- Run scripts from the project directory.
- For best results, use in a well-lit environment.

## License
MIT
