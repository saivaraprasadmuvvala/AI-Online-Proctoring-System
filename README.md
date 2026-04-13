# Realtime Online Proctoring System

A functional prototype of an AI-assisted real-time online proctoring system for educational use. This system monitors exam candidates through webcam video, detects suspicious behaviors, verifies identity, and logs anomalies.

## Features

- **Real-time Face Detection**: Uses MediaPipe for lightweight, CPU-friendly face detection
- **Identity Verification**: Primary method uses face_recognition embeddings, with SSIM fallback
- **Gaze Estimation**: Detects when candidates look away from screen
- **Tab/App Switch Detection**: Monitors browser tab switching and window focus using JavaScript events
- **Anomaly Detection**: Rule-based engine detects:
  - Face missing (>3 seconds)
  - Multiple faces detected
  - Identity mismatch
  - Off-screen gaze (>4 seconds)
  - Tab switching
  - Window/app switching
- **Instructor Dashboard**: Review sessions and anomalies with evidence images

## Technology Stack

- **Frontend**: Streamlit + streamlit-webrtc
- **Backend**: Python 3.12.0 (recommended) or 3.11.x
- **AI/ML**: MediaPipe, OpenCV, face_recognition (optional), SSIM
- **Database**: SQLite
- **Browser Events**: Page Visibility API, window.onblur/onfocus

## Installation (Windows)

### Step 1: Install Python
- Download and install **Python 3.12.0** (or 3.11.x) from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

### Step 2: Install Required Packages
Open Command Prompt or PowerShell in the project directory and run:
```bash
pip install -r requirements.txt
pip install git+https://github.com/ageitgey/face_recognition_models 
pip install ultralytics transformers torch torchvision
pip install pyaudio mss pyttsx3
```

```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.13
```

# Install PortAudio (system dependency)
```bash

brew install portaudio
# Then install pyaudio
pip install pyaudio
```

This will install:
- streamlit
- streamlit-webrtc
- opencv-python
- mediapipe
- numpy
- pillow
- scikit-image
- face-recognition

### Step 3: Create Required Directories
```bash
mkdir enrolled
mkdir evidence
```

### Step 4: Run the Application
```bash
streamlit run main.py
```

## Usage

1. **Start the application**:
   ```bash
   streamlit run main.py
   ```

2. **Enroll a candidate**:
   - Navigate to the Enrollment page
   - Enter candidate name
   - Capture reference face image using webcam
   - System will store enrollment data

3. **Start an exam session**:
   - Navigate to the Exam Session page
   - Enter candidate name and exam title
   - Click "Start Session"
   - System will begin real-time monitoring
   - Click "Stop Session" when done

4. **Review sessions**:
   - Navigate to the Instructor Dashboard
   - View all sessions
   - Click on a session to see detailed event timeline
   - Review evidence images
   - Mark sessions as reviewed

## Project Structure

```
online-exam-protectoring/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                    # Streamlit main entry point
├── modules/
│   ├── __init__.py
│   ├── detector.py           # MediaPipe face detection
│   ├── recognizer.py         # Identity verification
│   ├── gaze.py               # Gaze estimation
│   ├── anomaly.py            # Anomaly detection engine
│   ├── js_events.py          # Browser event handling
│   └── storage.py            # SQLite database wrapper
├── enrolled/                 # Enrollment images and embeddings
├── evidence/                 # Anomaly evidence images
└── static/                   # Static assets
```

## Event Types

The system detects and logs the following events:

- `face_missing`: No face detected for more than 3 seconds
- `multiple_faces`: More than one face detected in frame
- `identity_mismatch`: Detected face doesn't match enrolled candidate
- `gaze_offscreen`: User looking away for more than 4 seconds
- `tab_switch`: Browser tab was switched (Page Visibility API)
- `window_blur`: Window lost focus / user alt-tabbed to another app
- `window_focus`: Window regained focus

## Limitations

- This is a **functional prototype**, not production-ready
- SSIM identity matching is weaker than deep learning models
- Tab/app switch detection depends on browser JavaScript environment
- Performance may vary based on hardware (target: ≥10 FPS)
- Lighting conditions may affect detection accuracy

## License

This is a college-level functional prototype for educational purposes.

## Notes

- All data is stored locally (no cloud services)
- No external network calls are made
- Evidence images are saved with timestamps
- Database is SQLite (local file)
