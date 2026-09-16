# 📋 Product Requirements Document (PRD)

## Project: Squash Personal Trainer — AI-Powered Computer Vision & Generative Coaching System
**Document Version:** 1.2  
**Status:** Live / MVP Complete  
**Repository:** [github.com/soniasalmann/PersonalSquashTrainer](https://github.com/soniasalmann/PersonalSquashTrainer)  

---

## 1. Executive Summary

### 1.1 Mission Statement
Squash Personal Trainer democratizes elite sports coaching by transforming standard smartphone or match video footage into actionable, professional-grade biomechanical insights and personalized training programs.

### 1.2 The Core Value Proposition
> *"Computer vision measures what the player actually did; Generative AI turns those measurements into personalized coaching and training recommendations."*

Traditional sports coaching relies on expensive private instructors ($60–$150/hr) or subjective human observation. Squash Personal Trainer introduces a closed-loop system:
1. **Objective Computer Vision:** Tracks the player's 17 skeletal joints, tracks the high-speed ball with a Kalman filter, and projects movements onto real-world court coordinates ($6.4\text{m} \times 9.75\text{m}$) via planar homography.
2. **Generative AI Coaching Layer:** Feeds physical metrics (speed, distance, lunges, T-zone recovery %, distance per shot) into **Google Gemini**, producing structured tactical critiques and customized drill routines.

---

## 2. Problem Statement & Market Opportunity

### 2.1 The Problem
* **Squash Dynamics:** Squash is widely considered the world's most physically demanding racket sport. High-tempo rallies require rapid return to the central "T-Zone" and deep lunging without knee collapse.
* **Invisible Bottlenecks:** Solo practice (ghosting drills and solo hitting) lacks feedback. Players cannot accurately measure their recovery speed, court distance, or whether they are recovering to the T after striking.
* **Computer Vision Challenges:**
  * The squash ball is tiny (40mm), black, and moves at speeds up to 200+ km/h against dark floor lines and tin markings.
  * Monocular practice footage is taken from steep perspective angles, distorting real-world distances.

### 2.2 The Solution
An end-to-end automated pipeline that accepts raw practice video and generates:
1. An annotated video with skeleton overlays, ball trajectory tails, dynamic HUD, and top-down court minimap.
2. Objective session metrics (speed, distance covered, lunge count, T-recovery percentage).
3. A personalized GenAI coaching assessment and custom corrective drill plan rendered into interactive HTML dashboards, Markdown, and plain text.

---

## 3. User Personas & User Journeys

### Persona A: The Solo Competitive Player ("Hamza")
* **Goal:** Improve tournament endurance, movement economy, and T-recovery discipline.
* **Journey:** Records 10 minutes of solo practice on a phone tripod against the back glass wall. Feeds video into Squash Trainer. Views the interactive HTML report to discover his T-recovery is only 44%, and practices the prescribed corner-to-T ghosting drills.

### Persona B: The Club Coach ("Coach Sarah")
* **Goal:** Provide data-backed session reviews to students without manual video tagging.
* **Journey:** Runs student video through the pipeline. Exports the 10-second preview clip and the structured Markdown report to share on WhatsApp or training portals.

---

## 4. Product Architecture

```
Raw Video (.mp4 / .mov)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  PASS 1: Computer Vision & Kinematics Engine           │
│  - Player Pose: Ultralytics YOLOv8-Pose (17 Keypoints) │
│  - Ball Tracking: Custom YOLO Detector + Kalman Filter │
│  - Spatial Calibration: 3x3 Planar Homography Matrix   │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Objective Session Telemetry Compilation               │
│  - Total Distance (m)    - Avg / Peak Speed (km/h)     │
│  - Lunges Detected       - T-Recovery Rate (%)         │
│  - Shots Hit Count       - Distance / Shot (m)         │
└──────────┬─────────────────────────────────┬───────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────────┐       ┌────────────────────────┐
│  PASS 2: Video Engine  │       │  AI Squash Coach Layer │
│  - Skeleton & Trails   │       │  - Google Gemini Flash │
│  - Dynamic HUD Metrics │       │  - Strict JSON Schema  │
│  - 2D Minimap Overlay  │       │  - Offline Fallback    │
└──────────┬─────────────┘       └───────────┬────────────┘
           │                                 │
           ▼                                 ▼
  Processed Video (.mp4)           Session Reports (.html, .md, .txt)
```

---

## 5. Detailed Functional Specifications

### 5.1 Court Homography Calibration (`utils/court_calibrator.py`)
* Computes perspective transformation matrix $H$ using 4 known court floor corners:
  * Front-Left, Front-Right, Back-Left, Back-Right.
* Maps camera pixel space $(u, v)$ to international squash court standard dimensions: $X \in [0, 6.4\text{m}], Y \in [0, 9.75\text{m}]$.
* Configurable central "T-Zone" polygon for tactical recovery detection.

### 5.2 Ball State Estimation (`trackers/squash_ball_tracker.py`)
* Custom fine-tuned YOLOv8 weights trained on annotated squash footage (Roboflow & Google Colab).
* 2D Constant Velocity Kalman Filter:
  $$\mathbf{x}_k = [x, y, v_x, v_y]^T$$
* Tracks ball trajectory across occlusion or frame-skip gaps, smoothing noise and generating fading visual trail vectors.

### 5.3 Pose & Biomechanical Lunge Engine (`trackers/squash_player_tracker.py` & `utils/squash_analytics.py`)
* Tracks 17 skeletal keypoints (shoulders, hips, knees, ankles).
* **Lunge Detection:** Computes relative hip-to-ankle vertical displacement against torso length with temporal debouncing (1.2s) to prevent duplicate counts.
* **Kinematics Glitch Filter:** Suppresses keypoint teleportation artifacts by bounding maximum human running velocity to realistic physical limits ($\le 12.5\text{ m/s}$).

### 5.4 Video HUD & Minimap Renderer (`drawers/squash_drawers.py`)
* **Real-Time HUD:** Speedometer gauge, total distance counter, lunge counter, T-recovery indicator.
* **2D Minimap:** Renders a scaled 2D court floor in the top corner showing real-time player and ball positioning in court-space.

### 5.5 Generative AI Coaching Layer (`utils/squash_ai_coach.py`)
* **SDK:** Modern `google.genai` client with automatic fallback to `google.generativeai`.
* **Model Routing:** Targets active Flash models: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`.
* **Structured JSON Schema:**
  * `executive_summary`: Brief natural-language assessment.
  * `playing_style`: Player archetype based on speed and movement efficiency.
  * `strengths`: Bulleted tactical and physical highlights.
  * `areas_to_improve`: Primary areas requiring work.
  * `tactical_analysis`: T-zone craft and court positioning critique.
  * `movement_analysis`: Lunge quality and movement economy critique.
  * `training_plan`: Array of prescribed drills (`drill`, `duration`, `sets`, `focus`, `reason`).
  * `coach_summary`: Closing coach guidance.
* **Anti-Hallucination Guardrails:**
  * The model is strictly instructed to reason *only* from the supplied physical metrics.
  * Explicitly identified as "AI Squash Coach" (never claiming human certifications).
* **Zero-Failure Fallback Guarantee:**
  * If no API key is supplied, or on network timeout/error, a built-in deterministic heuristic engine generates the exact same structured schema offline.

---

## 6. Non-Functional Requirements (NFRs)

| Attribute | Specification |
| :--- | :--- |
| **Execution Performance** | Supports frame-skipping (`--skip-frames N`) and scaling (`--scale 0.5`) to allow fast processing on standard consumer laptops. |
| **GPU / CPU Portability** | Auto-detects CUDA GPU when available; runs cleanly on CPU without external driver dependencies. |
| **Security & Privacy** | API keys are read strictly from environment variables or runtime CLI args; never stored in code, HTML outputs, or git commits. |
| **Reliability** | Pipeline guaranteed never to crash due to LLM network drops or missing keys. |
| **Cross-Platform** | Fully operational on Windows 10/11, macOS, and Linux. |

---

## 7. Deliverables & Output Formats

1. **Annotated Video:** `output_videos/[name]_processed.mp4` with HUD and minimap.
2. **Preview Clip:** `output_videos/[name]_preview.mp4` (configurable 10-second preview clip).
3. **Interactive HTML Dashboard:** `output_videos/[name]_report.html` (responsive dark sports-analytics theme).
4. **Structured Markdown Report:** `output_videos/[name]_report.md` (separated into Objective Metrics vs AI Coach Insights).
5. **Text Summary Log:** `output_videos/[name]_report.txt`.

---

## 8. Product Roadmap

* [x] **v1.0 (Core Vision Engine):** YOLOv8-pose tracking, court homography, speed and lunge detection.
* [x] **v1.1 (Fast Mode & HUD):** Minimap overlay, frame skipping, preview video export.
* [x] **v1.2 (Generative AI Layer):** Google Gemini structured JSON coaching engine, offline fallback, HTML/MD dashboards.
* [ ] **v1.3 (Two-Player Match Mode):** Player separation, shot rally length tracking, and competitive unforced error analysis.
* [ ] **v1.4 (Acoustic Ball Impact Sync):** Racket impact detection via video audio waveform analysis.
* [ ] **v1.5 (Cloud / Web App):** Web interface for drag-and-drop video uploads and cloud GPU processing.
