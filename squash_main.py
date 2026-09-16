import cv2
import numpy as np
import json
import os
import argparse
import sys
import subprocess
import torch

from utils.court_calibrator import CourtCalibrator
from trackers.squash_player_tracker import SquashPlayerTracker
from trackers.squash_ball_tracker import SquashBallTracker
from utils.squash_analytics import SquashAnalytics
from drawers.squash_drawers import SquashDrawers
from utils.squash_ai_coach import (
    get_coaching_insights,
    build_markdown_report,
    build_html_report,
    build_text_report
)

def parse_args():
    parser = argparse.ArgumentParser(description="Squash Personal Trainer - Video Analysis Pipeline")
    parser.add_argument("video_path", type=str, help="Path to input squash practice video (.mp4)")
    parser.add_argument("--config", type=str, default="court_config.json", help="Path to court calibration config (.json)")
    parser.add_argument("--output", type=str, default=None, help="Path to output processed video (.mp4)")
    parser.add_argument("--ball_model", type=str, default="models/best.pt", help="Path to YOLO ball detection weights")
    parser.add_argument("--pose_model", type=str, default="models/yolov8m-pose.pt", help="Path to YOLO pose weights")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold for ball tracking")
    # Fast‑mode optional flags
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window (faster)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every (skip+1)th frame; 0 = every frame")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for input frames (e.g., 0.5 for half size)")
    parser.add_argument("--gpu", choices=["auto", "force", "cpu"], default="auto", help="GPU usage: auto (detect), force (require), cpu (disable)")
    parser.add_argument("--export-preview", action="store_true", help="Save a short preview clip (default 10s) for sharing")
    parser.add_argument("--preview-length", type=int, default=10, help="Length of preview video in seconds (used with --export-preview)")
    # Generative AI Coach flag
    parser.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key for GenAI coaching insights (or set GEMINI_API_KEY env var)")
    return parser.parse_args()

def transcode_to_h264(temp_path, output_path):
    """Transcodes the temp video to H.264 format using ffmpeg for browser compatibility."""
    print("\n[Video Engine] Transcoding output video to H.264...")
    transcoded = False
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", temp_path,
                "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                "-an", # Remove audio if any
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            os.remove(temp_path)
            transcoded = True
            print("[Video Engine] Transcoding complete!")
        else:
            print("[Video Engine] ffmpeg transcode failed. Keeping temporary video.")
            print(result.stderr.decode(errors="replace")[-500:])
    except Exception as e:
        print(f"[Video Engine] ffmpeg not available ({e}). Keeping temporary video.")
        
    if not transcoded:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

def generate_default_config(config_path):
    """Generates a template court configuration file if one does not exist."""
    template = {
        "back_wall_left": [100, 1000],
        "back_wall_right": [1820, 1000],
        "front_wall_left": [600, 300],
        "front_wall_right": [1320, 300],
        "notes": "Replace these with the actual pixel coordinates [x, y] of the four floor corners from your video."
    }
    with open(config_path, 'w') as f:
        json.dump(template, f, indent=4)
    print(f"\n[Court Calibrator] Created a template calibration file at '{config_path}'.")
    print("Please edit this file with your video's specific court corners before running again.")

def main():
    args = parse_args()
    
    # ---- GPU handling -----------------------------------------------------
    if args.gpu == "cpu":
        device = "cpu"
    elif args.gpu == "force":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU forced but CUDA not available")
        device = "cuda"
    else:  # auto
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.video_path):
        print(f"Error: Input video not found at '{args.video_path}'")
        sys.exit(1)

    # Output paths setup
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    os.makedirs("output_videos", exist_ok=True)
    
    # Store the preview length (seconds) for later use
    preview_len_seconds = args.preview_length if args.export_preview else 0
    
    if args.output is None:
        args.output = f"output_videos/output_{video_stem}_processed.mp4"
    temp_output_path = args.output.replace(".mp4", "_temp.mp4")

    # Calibration Setup
    if not os.path.exists(args.config):
        print(f"[Warning] Calibration config not found at '{args.config}'")
        generate_default_config(args.config)
        print("Using dummy fallback calibration for this run.")
        # Create a dummy config in memory so the pipeline doesn't crash
        calibrator = CourtCalibrator()
        calibrator.set_calibration_points([100, 1000], [1820, 1000], [600, 300], [1320, 300])
    else:
        calibrator = CourtCalibrator(args.config)
    # Extract court dimensions (width and length in meters) from calibrator
    court_w = float(calibrator.dst_points[1][0])  # width (meters)
    court_l = float(calibrator.dst_points[2][1])  # length (meters)

    # Initialize Trackers
    print("\n[Tracker Setup] Initializing models...")
    # Verify model weights are available, fall back to downloading coco ones if needed
    for model_file in [args.ball_model, args.pose_model]:
        if not os.path.exists(model_file) and "models/" in model_file:
            print(f"[Warning] Weights file not found: {model_file}")
            # Try to load standard ultralytics weights
            if "pose" in model_file:
                args.pose_model = "yolov8m-pose.pt"
            else:
                args.ball_model = "yolov8n.pt"

    # Open Video for Pass 1 – get fps before creating trackers
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total_frames = 100

    # Apply user‑scale to display size (only affects processing, not final output size)
    input_scale = args.scale if args.scale > 0 else 1.0

    # Initialize Trackers after fps is known
    player_tracker = SquashPlayerTracker(args.pose_model)
    ball_tracker = SquashBallTracker(args.ball_model, fps=fps)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n[Pass 1] Starting frame-by-frame detection...")
    print(f"Video specs: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")

    # Coordinate containers
    player_keypoints_history = []
    player_feet_pixels = []
    ball_centers_pixels = []
    ball_bboxes_pixels = []
    ball_is_predicted_history = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply scaling if requested
        if args.scale != 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * args.scale), int(frame.shape[0] * args.scale)))
        
        # Frame skipping logic
        if args.skip_frames > 0 and (frame_idx % (args.skip_frames + 1) != 0):
            frame_idx += 1
            continue

        # Progress bar
        pct = (frame_idx + 1) / total_frames * 100 if total_frames > 0 else 0
        sys.stdout.write(f"\rAnalyzing Frames: [{frame_idx + 1}/{total_frames}] {pct:.1f}%")
        sys.stdout.flush()

        # 1. Track Player
        pose_res = player_tracker.detect_frame(frame)
        kps, p_bbox = player_tracker.get_player_keypoints(pose_res)
        feet_pixel = player_tracker.get_player_feet_center(kps)
        
        player_keypoints_history.append(kps)
        player_feet_pixels.append(feet_pixel)

        # 2. Track Ball (uses Kalman filtering inside)
        ball_center, b_bbox, is_pred = ball_tracker.track_frame(frame, conf=args.conf)
        ball_centers_pixels.append(ball_center)
        ball_bboxes_pixels.append(b_bbox)
        ball_is_predicted_history.append(is_pred)

        frame_idx += 1

    cap.release()
    print("\n[Pass 1] Complete! Running calculations and smoothing filters...")

    # --- Calculations & Analytics ---
    # When frames are skipped, the real time-step between two processed frames is larger.
    # effective_fps = original_fps / (skip+1) so analytics gets the right dt.
    effective_fps = fps / (args.skip_frames + 1) if args.skip_frames > 0 else fps
    analytics = SquashAnalytics(fps=effective_fps, court_calibrator=calibrator, scale=input_scale)
    
    # Calculate Player Positions & Speed in Meters
    player_positions_m, player_speeds_kmh, total_distance = analytics.calculate_player_movement(player_feet_pixels)
    
    # Detect Lunges
    lunge_frames = analytics.detect_lunges(player_keypoints_history)
    print(f"[Analytics] Detected {len(lunge_frames)} lunges throughout the session.")

    # Calculate T-Recovery Rate
    # Transform ball pixels to meters for the T-Recovery check
    ball_positions_m = []
    for center in ball_centers_pixels:
        if center is not None:
            ball_positions_m.append(calibrator.pixel_to_court(center[0], center[1]))
        else:
            ball_positions_m.append(None)
            
    strikes, t_recovery_rate, recovery_details = analytics.analyze_t_recovery(player_positions_m, ball_centers_pixels, player_speeds_kmh)
    print(f"[Analytics] Detected {len(strikes)} ball strikes. T-Recovery Rate: {t_recovery_rate:.1f}%")
    print(f"[Analytics] Total distance covered: {total_distance:.2f} meters.")

    # --- Pass 2: Rendering and Output ---
    print("\n[Pass 2] Rendering overlays and writing output video...")
    cap = cv2.VideoCapture(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output frame size must match the (possibly scaled) frames we will write
    out_w = int(width * input_scale)
    out_h = int(height * input_scale)
    out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
    
    drawers = SquashDrawers()
    
    # Active lunge counter trace
    current_lunges_count = 0
    lunge_set = set(lunge_frames)
    
    # Player heatmap trail (meters)
    heatmap_trail = []

    # Prepare list for optional preview export
    preview_frames = []
    max_preview_frames = int(preview_len_seconds * fps) if args.export_preview else 0

    frame_idx = 0
    data_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply scaling if user requested (same as Pass 1)
        if args.scale != 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * args.scale), int(frame.shape[0] * args.scale)))
        
        # Frame skipping (use same logic as Pass 1)
        if args.skip_frames > 0 and (frame_idx % (args.skip_frames + 1) != 0):
            frame_idx += 1
            continue
        
        # At this point we have a frame that corresponds to an entry in the data arrays
        if data_idx >= len(player_positions_m):
            break
        
        # Progress bar
        pct = (frame_idx + 1) / total_frames * 100 if total_frames > 0 else 0
        sys.stdout.write(f"\rDrawing Overlays: [{frame_idx + 1}/{total_frames}] {pct:.1f}%")
        sys.stdout.flush()
        
        # Update lunge counts dynamically.
        # lunge_frames contains DATA indices (0,1,2,...) not raw frame indices,
        # so compare against data_idx.
        if data_idx in lunge_set:
            current_lunges_count += 1
        
        # Maintain a rolling trail of the player's 2D position for the heatmap (last 120 frames / 4 seconds)
        p_pos = player_positions_m[data_idx]
        if p_pos is not None:
            heatmap_trail.append(p_pos)
            if len(heatmap_trail) > 120:
                heatmap_trail.pop(0)
        
        # 1. Draw Player Pose & Knee Angles
        kps = player_keypoints_history[data_idx]
        knee_angles = player_tracker.get_knee_angles(kps)
        frame = drawers.draw_player_pose(frame, kps, knee_angles)
        
        # 2. Draw Ball Trajectory
        frame = drawers.draw_ball_trajectory(frame, ball_centers_pixels, data_idx, tail_len=15)
        
        # 3. Draw Top-down Minimap HUD
        b_pos = ball_positions_m[data_idx]
        frame = drawers.draw_minimap(frame, p_pos, b_pos, heatmap_trail, court_w=court_w, court_l=court_l)
        
        # 4. Draw HUD Metrics overlay
        speed = player_speeds_kmh[data_idx]
        # Pass effective_fps so the timer shows real wall-clock seconds,
        # not data-frame count divided by original fps.
        frame = drawers.draw_hud_metrics(frame, speed, current_lunges_count, t_recovery_rate, data_idx, effective_fps)
        
        # Conditional live preview
        if not args.no_preview:
            cv2.imshow("Squash Personal Trainer - Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User Interrupt] Stopped playback.")
                break
        
        out_writer.write(frame)
        
        # Capture for preview export (store first N frames)
        if args.export_preview and len(preview_frames) < max_preview_frames:
            preview_frames.append(frame.copy())
        
        frame_idx += 1
        data_idx += 1

    cap.release()
    out_writer.release()
    if not args.no_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("\n[Pass 2] Complete!")
    
    # Write preview video if requested
    if args.export_preview and preview_frames:
        preview_path = args.output.replace('.mp4', '_preview.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (out_w, out_h))
        for pf in preview_frames:
            preview_writer.write(pf)
        preview_writer.release()
        print(f"[Preview Export] Saved short preview clip to {preview_path}")

    # Transcode temp mp4v output to H.264
    transcode_to_h264(temp_output_path, args.output)

    # Save Session Summary Reports (Markdown, HTML, and Text formats)
    report_path = args.output.replace(".mp4", "_report.txt")
    md_report_path = args.output.replace(".mp4", "_report.md")
    html_report_path = args.output.replace(".mp4", "_report.html")
    
    avg_speed = float(np.mean(player_speeds_kmh)) if len(player_speeds_kmh) > 0 else 0.0
    peak_speed = float(np.max(player_speeds_kmh)) if len(player_speeds_kmh) > 0 else 0.0
    num_lunges = len(lunge_frames)
    num_strikes = len(strikes)
    duration_sec = total_frames / fps
    dist_per_shot = total_distance / max(1, num_strikes)

    # 1. Compile objective session metrics dictionary
    session_metrics = {
        "duration_sec": duration_sec,
        "total_distance_m": float(total_distance),
        "avg_speed_kmh": avg_speed,
        "peak_speed_kmh": peak_speed,
        "num_lunges": num_lunges,
        "num_strikes": num_strikes,
        "dist_per_shot": float(dist_per_shot),
        "t_recovery_rate": float(t_recovery_rate)
    }

    # 2. Query AI Squash Coach (Google Gemini with seamless rule-based fallback)
    coaching_result = get_coaching_insights(session_metrics, api_key=args.gemini_key)

    # 3. Write plain text report
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write(build_text_report(session_metrics, coaching_result, args.video_path))

    # 4. Write structured Markdown report
    with open(md_report_path, 'w', encoding='utf-8') as rf:
        rf.write(build_markdown_report(session_metrics, coaching_result, args.video_path))

    # 5. Write interactive HTML5 report
    with open(html_report_path, 'w', encoding='utf-8') as rf:
        rf.write(build_html_report(session_metrics, coaching_result, args.video_path))

    print(f"\n[Pipeline Finished] Successfully completed analysis!")
    print(f"Processed Video Saved: {args.output}")
    print(f"Text Summary Saved:     {report_path}")
    print(f"Interactive MD Report:  {md_report_path}")
    print(f"Interactive HTML Report:{html_report_path}")

if __name__ == "__main__":
    main()
