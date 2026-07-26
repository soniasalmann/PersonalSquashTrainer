# 🎾 Squash Trainer — AI-Powered Personal Squash Coach & Analytics System

An end-to-end computer vision and deep learning pipeline that transforms raw squash practice or match footage into real-world biomechanical insights and tactical coaching feedback.

Squash Trainer acts as an automated personal coach, calculating player speed (km/h), distance covered, lunge counts, and T-recovery metrics. The system calibrates camera perspective angles using planar homography, tracks movements with advanced state estimation, and generates visual HUD overlays alongside top-down court positioning minimaps.

---

## ✨ Key Features

* 📐 **Court Homography Calibration:** Solves $3 \times 3$ perspective transformation matrices to map 2D camera pixels directly to standard real-world squash court coordinates ($6.4\text{m} \times 9.75\text{m}$).
* 🏃 **Speed & Distance Engine:** Calculates calibrated movement velocity (km/h) and total distance covered, applying scale correction and kinematics glitch filtering.
* 🦵 **Automated Pose & Lunge Detection:** Monitors 17 keypoint joint coordinates frame-by-frame using deep pose models to evaluate footwork intensity and count lunges.
* 🎯 **T-Zone Recovery Tracking:** Automatically evaluates player recovery rates by measuring how consistently and quickly the player returns to the central T-zone after executing shots.
* 🗺️ **Live HUD Overlay & Minimap:** Renders real-time metrics, fading ball trajectory paths, player tracking trails, and a top-down court position map onto the output video.
* 📊 **Coaching Suggestions Report:** Generates clean post-session HTML dashboards and text summaries with targeted, data-driven coaching recommendations.

---

## 🧠 Custom Model Training & Data Pipeline

Standard off-the-shelf object detection models often fail to track a tiny, high-speed black squash ball against dark court lines. To solve this, this project incorporates custom ML model training:
1. **Dataset Annotation:** Curated and annotated a custom squash ball and court dataset using **Roboflow**.
2. **Model Training:** Trained and fine-tuned custom YOLOv8 detection weights using PyTorch on **Google Colab** GPUs.
3. **Tracking & State Estimation:** Combined the custom detection weights with a 2D constant velocity **Kalman Filter** state-space estimator ($[x, y, v_x, v_y]$) to predict ball trajectory paths continuously across frame gaps.

---

## 🛠️ Tech Stack

* **Computer Vision & AI:** Ultralytics YOLOv8-Pose (Player Tracking), Custom YOLO Ball Detector, PyTorch
* **Model Training & Data Pipeline:** Google Colab (GPU Acceleration), Roboflow (Custom Dataset)
* **Kinematics & Math:** Planar Homography Transformation, Kalman Filtering, NumPy, SciPy
* **Video Engine & HUD:** OpenCV Video Processing, Dynamic HUD & Court Minimap Overlay
* **Analytics & Reporting:** Custom HTML5/CSS3 Dashboards, Markdown Generators

---

## 📁 Folder Structure

```
squashtrainer/
├── squash_main.py            # Main entrypoint running the 2-pass streaming pipeline
├── court_config.json         # Court line pixel calibration coordinates configuration
├── calibrate_court.py        # Utility to interactively find court coordinates
├── trackers/
│   ├── squash_ball_tracker.py   # Custom Kalman filter-based ball tracking
│   └── squash_player_tracker.py # Pose tracking and joint angle calculations
├── drawers/
│   └── squash_drawers.py        # Renders player pose, ball tails, HUD, and 2D court minimap
├── utils/
│   ├── court_calibrator.py      # Spatial coordinate homography mappings (pixels -> meters)
│   └── squash_analytics.py      # Speed, distance, lunge, and T-recovery calculations
└── output_videos/               # Processed videos, HTML reports, and summaries (auto-generated)
```

---

## 🚀 Getting Started

### 1. Requirements

Ensure you have the required packages installed in your Python environment:
```bash
pip install opencv-python ultralytics torch numpy scipy
```

### 2. Model Weights

Create a `models/` folder in the project directory and place your weights files there:
* `models/yolov8m-pose.pt` (Pose model)
* `models/best.pt` (Custom fine-tuned ball detector model)

*Note: If the `models/` folder or weights are missing, the pipeline will fall back to downloading and using default weights.*

### 3. Usage

Run the main pipeline script from your terminal:
```bash
python squash_main.py "path/to/your/input_video.mp4" --config court_config.json --output "output_videos/processed_output.mp4" --no-preview --skip-frames 2 --scale 0.5 --export-preview --preview-length 10
```

#### Command Line Arguments:
* `video_path` (Position 1): Path to the input video.
* `--config`: Path to the court line calibration coordinates json file (default: `court_config.json`).
* `--output`: Path to save the processed output video.
* `--no-preview`: Suppress real-time playback window (recommended for headless execution/speed).
* `--skip-frames`: Skip frames to accelerate processing (e.g. `2` skips every 2 of 3 frames, processing at 1/3 frame rate).
* `--scale`: Rescale input video resolution (e.g., `0.5` for 50% scale processing).
* `--export-preview`: Exports a short preview clip of the output.
* `--preview-length`: Duration of the exported preview clip in seconds (default: `10`).

---

## 📊 Session Analytics Reports

After the pipeline completes processing, it saves three summary report files detailing session performance:
1. **Interactive HTML Report:** `output_videos/processed_output_report.html` — A styled, dark-themed dashboard summarizing workout metrics, tactical stats, and automated coach recommendations.
2. **Markdown Report:** `output_videos/processed_output_report.md` — A structured markdown file containing metrics and suggestions.
3. **Text Summary:** `output_videos/processed_output_report.txt` — A clean plain-text log.
