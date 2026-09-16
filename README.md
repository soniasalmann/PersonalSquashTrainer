# 🎾 Squash Trainer — AI-Powered Personal Squash Coach & Analytics System

An end-to-end computer vision and deep learning pipeline that transforms raw squash practice or match footage into real-world biomechanical insights and tactical coaching feedback.

Squash Trainer acts as an automated personal coach, calculating player speed (km/h), distance covered, lunge counts, and T-recovery metrics. The system calibrates camera perspective angles using planar homography, tracks movements with advanced state estimation, renders live HUD video overlays, and leverages a **Generative AI Coach Layer (Google Gemini)** to turn physical measurements into targeted, personalized training programs.

> 💡 **Core Principle:** *"Computer vision measures what the player actually did; Generative AI turns those measurements into personalized coaching and training recommendations."*

---

## 🏛️ Pipeline Architecture

```
Raw Squash Video (.mp4)
         ↓
Computer Vision Engine (Ultralytics YOLOv8-Pose + Custom Ball Detector)
         ↓
Kinematics & Spatial Geometry (Planar Homography 6.4m × 9.75m + 2D Kalman Filter)
         ↓
Objective Session Telemetry Dictionary (Speed, Distance, Lunges, T-Recovery %, Dist/Shot)
         ↓
AI Squash Coach (Google Gemini 2.5 Flash with Strict Structured JSON Schema)
         ↓  [Automatic Fallback to Offline Heuristic Engine if offline/keyless]
Comprehensive Multi-Format Session Dashboards (.html, .md, .txt)
```

---

## ✨ Key Features

* 📐 **Court Homography Calibration:** Solves $3 \times 3$ perspective transformation matrices to map 2D camera pixels directly to standard real-world squash court coordinates ($6.4\text{m} \times 9.75\text{m}$).
* 🏃 **Speed & Distance Engine:** Calculates calibrated movement velocity (km/h) and total distance covered, applying scale correction and kinematics glitch filtering.
* 🦵 **Automated Pose & Lunge Detection:** Monitors 17 keypoint joint coordinates frame-by-frame using deep pose models to evaluate footwork intensity and count lunges.
* 🎯 **T-Zone Recovery Tracking:** Automatically evaluates player recovery rates by measuring how consistently and quickly the player returns to the central T-zone after executing shots.
* 🗺️ **Live HUD Overlay & Minimap:** Renders real-time metrics, fading ball trajectory paths, player tracking trails, and a top-down court position map onto the output video.
* 🤖 **Generative AI Coach Layer (Google Gemini):** Synthesizes objective session measurements into actionable tactical feedback, strengths/weaknesses breakdowns, playing style archetypes, and custom prescribed training drills.
* 🛡️ **Zero-Failure Fallback Guarantee:** If `GEMINI_API_KEY` is not provided, or in cases of network drops/timeouts, the pipeline automatically falls back to deterministic rule-based coaching heuristics so processing never crashes.
* 📊 **Multi-Format Reports:** Generates interactive dark-themed HTML dashboards, structured Markdown reports, and plain-text summaries.

---

## 🧠 Custom Model Training & Data Pipeline

Standard off-the-shelf object detection models often fail to track a tiny, high-speed black squash ball against dark court lines. To solve this, this project incorporates custom ML model training:
1. **Dataset Annotation:** Curated and annotated a custom squash ball and court dataset using **Roboflow**.
2. **Model Training:** Trained and fine-tuned custom YOLOv8 detection weights using PyTorch on **Google Colab** GPUs.
3. **Tracking & State Estimation:** Combined the custom detection weights with a 2D constant velocity **Kalman Filter** state-space estimator ($[x, y, v_x, v_y]$) to predict ball trajectory paths continuously across frame gaps.

---

## 🛠️ Tech Stack

* **Computer Vision & AI:** Ultralytics YOLOv8-Pose (Player Tracking), Custom YOLO Ball Detector, PyTorch
* **Generative AI Layer:** Google Gemini Flash (`google-genai` / `google-generativeai`), Structured JSON Schema Decoding
* **Model Training & Data Pipeline:** Google Colab (GPU Acceleration), Roboflow (Custom Dataset)
* **Kinematics & Math:** Planar Homography Transformation, Kalman Filtering, NumPy, SciPy
* **Video Engine & HUD:** OpenCV Video Processing, Dynamic HUD & Court Minimap Overlay
* **Analytics & Reporting:** HTML5/CSS3 Interactive Dashboards, Markdown Generators

---

## 📁 Folder Structure

```
squashtrainer/
├── squash_main.py            # Main entrypoint running the 2-pass streaming pipeline
├── court_config.json         # Court line pixel calibration coordinates configuration
├── calibrate_court.py        # Utility to interactively find court coordinates
├── requirements.txt          # Python package dependencies
├── trackers/
│   ├── squash_ball_tracker.py   # Custom Kalman filter-based ball tracking
│   └── squash_player_tracker.py # Pose tracking and joint angle calculations
├── drawers/
│   └── squash_drawers.py        # Renders player pose, ball tails, HUD, and 2D court minimap
├── utils/
│   ├── court_calibrator.py      # Spatial coordinate homography mappings (pixels -> meters)
│   ├── squash_analytics.py      # Speed, distance, lunge, and T-recovery calculations
│   └── squash_ai_coach.py       # GenAI Coach (Gemini JSON engine + fallback + report formatters)
└── output_videos/               # Processed videos, HTML reports, and summaries (auto-generated)
```

---

## 🚀 Getting Started

### 1. Requirements & Installation

Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 2. Model Weights

Create a `models/` folder in the project directory and place your weights files there:
* `models/yolov8m-pose.pt` (Pose model)
* `models/best.pt` (Custom fine-tuned ball detector model)

*Note: If the `models/` folder or weights are missing, the pipeline will fall back to downloading and using default weights.*

### 3. Google Gemini Setup (Optional for GenAI Coach)

To enable the Generative AI coaching layer, set your Gemini API key in your environment:
```powershell
$env:GEMINI_API_KEY="your-google-gemini-api-key"
```
Or pass it directly via `--gemini-key "your-key"`.

*If omitted, the system seamlessly runs offline using the built-in deterministic heuristic coaching engine.*

### 4. Running the Pipeline

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
* `--gemini-key`: Google Gemini API key (optional; defaults to `GEMINI_API_KEY` environment variable).

---

## 📊 Session Analytics Reports

After the pipeline completes processing, it saves three summary report files detailing session performance:
1. **Interactive HTML Report:** `output_videos/processed_output_report.html` — A responsive dashboard with metric cards, style classification, strengths, weaknesses, and custom drill plans.
2. **Structured Markdown Report:** `output_videos/processed_output_report.md` — Clearly divides **Part 1: Objective CV Metrics** and **Part 2: AI Squash Coach Insights**.
3. **Text Summary:** `output_videos/processed_output_report.txt` — Plain-text session log.

### Example AI Coaching Output (Structured Training Plan)

| Drill Name | Duration | Sets / Reps | Target Focus | Data-Driven Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Ghosting: Deep Corner to T-Split Step** | `12 minutes` | `3 sets x 8 reps` | Explosive first recovery step back to T | *Observed T-recovery rate of 44.4% indicates delay recovering from corners.* |
| **Early Preparation & Linear Footwork Routine** | `12 minutes` | `3 sets x 6 minutes` | Direct diagonal lines & early racket prep | *Player averaged 5.58m per shot, signaling inefficient travel paths.* |
