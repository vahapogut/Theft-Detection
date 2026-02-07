# Theft Guard AI - Advanced Anti-Theft System

Theft Guard AI is a comprehensive security solution designed for retail environments. It leverages Computer Vision and Artificial Intelligence to detect suspicious behaviors such as shoplifting, fighting, and item concealment in real-time. The system processes video feeds from multiple CCTV cameras and provides instant alerts to security personnel via a modern web dashboard, Email, and Telegram.

## System Capabilities

### 1. Advanced Theft Detection
The core of the system is built on a multi-stage AI pipeline that runs locally:
*   **Object Detection:** Utilizes optimized YOLOv8 models to identify person and retail objects. It can be extended with specialized models ('shoplifting.pt') to directly detect theft actions.
*   **Behavior Analysis:** Tracks the movement of hands relative to pockets/bags to detect concealment attempts (Hand-to-Pocket gestures).
*   **Pose Estimation:** Analyzes human skeleton points to detect suspicious postures, such as bending down in aisles or reaching into restricted areas.

### 2. Facial Recognition System
*   **Blacklist Monitoring:** Instantly identifies known offenders entered into the system database and triggers high-priority alerts.
*   **VIP Detection:** Can be configured to recognize loyal customers or VIPs for personalized service.

### 3. Real-Time Surveillance Dashboard
built with Next.js, the dashboard offers a centralized control room experience:
*   **Live Video Feeds:** Stream vertically or horizontally from multiple camera sources (USB or IP Cameras) simultaneously via WebSockets.
*   **Dynamic Graphs:** Visualizes weekly and daily alert statistics to track security trends over time.
*   **Visual Alerts:** Flashing on-screen notifications highlight the camera and timestamp of any detected event.



### 4. Alert History & Evidence
*   **Event Logging:** Every detection is saved to a local SQLite database with a timestamp, alert type, and confidence score.
*   **Snapshot Capture:** High-resolution images of the event are automatically saved for evidence.
*   **History Viewer:** Browse past alerts, view snapshots, and export data directly from the interface.



### 5. Remote Notifications
Stay informed even when away from the desk:
*   **Telegram Integration:** Sends an instant photo and caption to a specified Telegram Chat ID via Bot API.
*   **Email Reports:** Dispatches detailed text alerts to configured email addresses using SMTP.

### 6. Customizable Security Zones
*   **Region of Interest (ROI):** Users can draw custom polygons on the camera feed to define sensitive areas (e.g., cash registers, high-value shelves).
*   **Zone-Specific Rules:** Detection sensitivity can be adjusted based on whether a person is inside or outside these zones.



## Technical Architecture

*   **Backend:** Python, FastAPI, OpenCV, Ultralytics YOLOv8, Face Recognition, Albumentations
*   **Frontend:** Next.js 14, React, Tailwind CSS, Recharts, Lucide React
*   **Database:** SQLite (Lightweight, local storage for events and faces)
*   **Communication:** WebSockets (Real-time data), SMTP (Email), HTTPS (Telegram API)

## Installation Guide

### Prerequisites
*   Python 3.9 or higher
*   Node.js (LTS version)
*   NVIDIA GPU with CUDA (Recommended for real-time performance)

### 1. Backend Configuration
Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Start-Up-Vahap/Theft-Detection.git
cd Theft-Detection
pip install -r requirements.txt
```

If you have a specialized model (shoplifting.pt), place it in the root directory. Otherwise, the system defaults to the standard YOLOv8 model with behavior logic.

### 2. Dashboard Setup
Navigate to the dashboard directory and install dependencies:

```bash
cd dashboard
npm install
```

## Usage

### Auto-Start
Simply run the helper script to launch both services:
```bash
start_system.bat
```

### Manual Startup
**Start Backend API:**
```bash
py backend.py
```

**Start Frontend UI:**
```bash
cd dashboard
npm run dev
```

Open your browser and navigate to `http://localhost:3000`.

## License
This project is open-source and available under the MIT License.

____________________________________________________________________

Developer : Abdulvahap Öğüt
