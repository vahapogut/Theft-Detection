# 🛡️ Theft Guard AI - Advanced Anti-Theft System

**Theft Guard AI** is a cutting-edge security solution powered by Computer Vision and Artificial Intelligence. It detects suspicious behaviors (shoplifting, fighting, concealment) in real-time using CCTV cameras and instantly alerting security personnel via Email or Telegram.

![Dashboard Overview](assets/screenshots/dashboard_1.png)
*(Place your dashboard screenshot here)*

## 🌟 Key Features

*   **🧠 Advanced AI Detection:**
    *   **Shoplifting Detection:** Uses specialized YOLOv8 models to detect theft actions.
    *   **Pose Analysis:** Analyses skeleton points (Pose Estimation) to detect "Reaching" or "Bending" anomalies.
    *   **Concealment Tracking:** Tracks "Hand-to-Pocket" movements to identify hidden items.
*   **👤 Facial Recognition:** Identifies VIPs or Blacklisted individuals instantly.
*   **📊 Real-Time Dashboard:**
    *   Live Video Feeds from multiple cameras.
    *   Weekly Activity Graphs & Daily Event Counters.
    *   Instant Visual Alerts on the screen.
*   **🔔 Instant Notifications:**
    *   **Telegram:** Receive snapshots and alerts directly to your phone.
    *   **Email:** Detailed incident reports with images.
*   **🔧 Easy Configuration:**
    *   Add/Remove cameras dynamically.
    *   Draw "Forbidden Zones" (ROI) on the screen.
    *   Toggle heatmap and sensitivity settings.

## 🚀 Installation

### Prerequisites
*   Python 3.9+
*   Node.js & npm
*   CUDA-enabled GPU (Recommended for faster inference)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Start-Up-Vahap/Theft-Detection.git
cd Theft-Detection

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Download specialized model
# Place 'shoplifting.pt' in the root directory. 
# Defaults to 'yolov8n.pt' with smart filtering if not found.
```

### 2. Frontend Setup
```bash
cd dashboard
npm install
```

## 💻 Usage

### One-Click Start
Run the `start_system.bat` file. It will automatically:
1.  Start the Python Backend API.
2.  Launch the Next.js Dashboard.
3.  Open your browser to the control panel.

### Manual Start
**Backend:**
```bash
py backend.py
```

**Frontend:**
```bash
cd dashboard
npm run dev
```

## 📸 Screenshots

| Dashboard | Alert History |
|-----------|---------------|
| ![Dashboard](assets/screenshots/dashboard_1.png) | ![Alerts](assets/screenshots/dashboard_2.png) |

| Settings | Mobile View |
|----------|-------------|
| ![Settings](assets/screenshots/settings.png) | ![Mobile](assets/screenshots/mobile.png) |

## 🛠️ Tech Stack
*   **Core:** Python, OpenCV, YOLOv8 (Ultralytics), Face Recognition
*   **Backend:** FastAPI, WebSockets, SQLite
*   **Frontend:** Next.js 14, React, Tailwind CSS, Recharts
*   **Communication:** SMTP (Email), Telegram Bot API

## 📜 License
This project is licensed under the MIT License.
