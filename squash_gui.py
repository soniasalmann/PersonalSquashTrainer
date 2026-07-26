import os
import sys
import json
import cv2
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

# Import existing core modules
from utils.court_calibrator import CourtCalibrator
from trackers.squash_player_tracker import SquashPlayerTracker
from trackers.squash_ball_tracker import SquashBallTracker
from utils.squash_analytics import SquashAnalytics
from drawers.squash_drawers import SquashDrawers

class SquashTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Squash Personal Trainer AI")
        self.root.geometry("1280x750")
        self.root.configure(bg="#121212")

        self.video_path = "input_videos/practice.mp4"
        self.config_path = "court_config.json"
        self.is_processing = False
        self.cap = None

        self.setup_styles()
        self.build_ui()

        # Auto-start analysis on launch after 500ms
        self.root.after(500, self.start_analysis_thread)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#121212")
        self.style.configure("Card.TFrame", background="#1e1e1e", relief="flat")
        self.style.configure("Header.TLabel", background="#121212", foreground="#00e676", font=("Segoe UI", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        self.style.configure("Body.TLabel", background="#1e1e1e", foreground="#cccccc", font=("Segoe UI", 10))
        self.style.configure("StatVal.TLabel", background="#1e1e1e", foreground="#00e676", font=("Segoe UI", 14, "bold"))
        self.style.configure("Accent.TButton", background="#00e676", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.map("Accent.TButton", background=[("active", "#00c853")])

    def build_ui(self):
        # 1. Top Bar / Header
        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(fill="x", side="top")

        lbl_title = ttk.Label(top_bar, text="🏸 SQUASH PERSONAL TRAINER AI", style="Header.TLabel")
        lbl_title.pack(side="left", padx=10)

        btn_select = ttk.Button(top_bar, text="📁 Select Video", command=self.select_video)
        btn_select.pack(side="left", padx=10)

        self.lbl_file = ttk.Label(top_bar, text=os.path.basename(self.video_path), background="#121212", foreground="#888888", font=("Segoe UI", 10))
        self.lbl_file.pack(side="left", padx=5)

        self.btn_run = ttk.Button(top_bar, text="▶ Re-run Analysis", style="Accent.TButton", command=self.start_analysis_thread)
        self.btn_run.pack(side="right", padx=10)

        btn_calib = ttk.Button(top_bar, text="🎯 Calibrate Court", command=self.launch_calibrator)
        btn_calib.pack(side="right", padx=5)

        # 2. Main Content Split (Left: Video Canvas, Right: Dashboard)
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill="both", expand=True, side="top")

        # Video Panel (Left)
        video_card = ttk.Frame(content_frame, style="Card.TFrame", padding=10)
        video_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_status = ttk.Label(video_card, text="Status: Initializing Analysis...", style="SubHeader.TLabel")
        self.lbl_status.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(video_card, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.progress = ttk.Progressbar(video_card, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(10, 0))

        # Dashboard Panel (Right)
        dash_card = ttk.Frame(content_frame, style="Card.TFrame", padding=15, width=380)
        dash_card.pack(side="right", fill="y", expand=False)
        dash_card.pack_propagate(False)

        lbl_dash_title = ttk.Label(dash_card, text="📊 Performance & Coaching", style="SubHeader.TLabel")
        lbl_dash_title.pack(anchor="w", pady=(0, 15))

        # Metrics Grid Cards
        grid_frame = ttk.Frame(dash_card, style="Card.TFrame")
        grid_frame.pack(fill="x", pady=(0, 15))

        # Metric 1: Distance
        m1 = ttk.Frame(grid_frame, style="Card.TFrame")
        m1.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(m1, text="Distance Covered", style="Body.TLabel").pack(anchor="w")
        self.lbl_val_dist = ttk.Label(m1, text="0.0 m", style="StatVal.TLabel")
        self.lbl_val_dist.pack(anchor="w")

        # Metric 2: Speed
        m2 = ttk.Frame(grid_frame, style="Card.TFrame")
        m2.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(m2, text="Peak Speed", style="Body.TLabel").pack(anchor="w")
        self.lbl_val_speed = ttk.Label(m2, text="0.0 km/h", style="StatVal.TLabel")
        self.lbl_val_speed.pack(anchor="w")

        # Metric 3: Lunges
        m3 = ttk.Frame(grid_frame, style="Card.TFrame")
        m3.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(m3, text="Lunges Count", style="Body.TLabel").pack(anchor="w")
        self.lbl_val_lunges = ttk.Label(m3, text="0", style="StatVal.TLabel")
        self.lbl_val_lunges.pack(anchor="w")

        # Metric 4: T-Recovery Rate
        m4 = ttk.Frame(grid_frame, style="Card.TFrame")
        m4.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(m4, text="T-Recovery Rate", style="Body.TLabel").pack(anchor="w")
        self.lbl_val_t_rate = ttk.Label(m4, text="0.0%", style="StatVal.TLabel")
        self.lbl_val_t_rate.pack(anchor="w")

        ttk.Separator(dash_card, orient="horizontal").pack(fill="x", pady=10)

        # Coaching Tips Section
        ttk.Label(dash_card, text="💡 AI Coach Recommendations", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

        self.txt_tips = tk.Text(dash_card, bg="#181818", fg="#dddddd", font=("Segoe UI", 9),
                                wrap="word", relief="flat", highlightthickness=0)
        self.txt_tips.pack(fill="both", expand=True)
        self.txt_tips.insert("1.0", "Analyzing session... Please wait.")
        self.txt_tips.config(state="disabled")

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Squash Practice Video", filetypes=[("MP4 Videos", "*.mp4"), ("All Files", "*.*")])
        if file_path:
            self.video_path = file_path
            self.lbl_file.config(text=os.path.basename(self.video_path))
            self.start_analysis_thread()

    def launch_calibrator(self):
        os.system(f'"{sys.executable}" calibrate_court.py')

    def start_analysis_thread(self):
        if self.is_processing:
            return
        if not os.path.exists(self.video_path):
            messagebox.showerror("Error", f"Video file not found: {self.video_path}")
            return

        self.is_processing = True
        self.btn_run.config(state="disabled")
        self.progress["value"] = 0
        self.lbl_status.config(text="Status: Pass 1/2 — AI Pose & Ball Tracking...")

        thread = threading.Thread(target=self.run_pipeline, daemon=True)
        thread.start()

    def run_pipeline(self):
        try:
            calibrator = CourtCalibrator(self.config_path) if os.path.exists(self.config_path) else CourtCalibrator()
            if not os.path.exists(self.config_path):
                calibrator.set_calibration_points([100, 1000], [1820, 1000], [600, 300], [1320, 300])

            pose_model = "models/yolov8m-pose.pt" if os.path.exists("models/yolov8m-pose.pt") else "yolov8m-pose.pt"
            ball_model = "models/best.pt" if os.path.exists("models/best.pt") else "yolov8n.pt"

            player_tracker = SquashPlayerTracker(pose_model)
            ball_tracker = SquashBallTracker(ball_model)

            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

            # Pass 1: Tracking
            player_keypoints_history = []
            player_feet_pixels = []
            ball_centers_pixels = []

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pose_res = player_tracker.detect_frame(frame)
                kps, _ = player_tracker.get_player_keypoints(pose_res)
                feet_pixel = player_tracker.get_player_feet_center(kps)

                player_keypoints_history.append(kps)
                player_feet_pixels.append(feet_pixel)

                ball_center, _, _ = ball_tracker.track_frame(frame, conf=0.15)
                ball_centers_pixels.append(ball_center)

                frame_idx += 1
                self.progress["value"] = (frame_idx / total_frames) * 50

            cap.release()

            # Analytics
            analytics = SquashAnalytics(fps=fps, court_calibrator=calibrator)
            player_positions_m, player_speeds_kmh, total_distance = analytics.calculate_player_movement(player_feet_pixels)
            lunge_frames = analytics.detect_lunges(player_keypoints_history)

            ball_positions_m = []
            for center in ball_centers_pixels:
                if center is not None:
                    ball_positions_m.append(calibrator.pixel_to_court(center[0], center[1]))
                else:
                    ball_positions_m.append(None)

            strikes, t_recovery_rate, _ = analytics.analyze_t_recovery(player_positions_m, ball_centers_pixels)

            # Update Dashboard Metrics
            avg_speed = float(np.mean(player_speeds_kmh)) if len(player_speeds_kmh) > 0 else 0.0
            peak_speed = float(np.max(player_speeds_kmh)) if len(player_speeds_kmh) > 0 else 0.0
            num_lunges = len(lunge_frames)
            num_strikes = len(strikes)

            self.root.after(0, self.update_dashboard_metrics, total_distance, peak_speed, num_lunges, t_recovery_rate)
            self.root.after(0, self.update_coaching_tips, t_recovery_rate, num_strikes, total_distance, num_lunges)
            self.root.after(0, lambda: self.lbl_status.config(text="Status: Pass 2/2 — Playing Annotated Stream Live..."))

            # Pass 2: Rendering and live smooth video playback inside canvas
            cap = cv2.VideoCapture(self.video_path)
            drawers = SquashDrawers()
            lunge_set = set(lunge_frames)
            current_lunges = 0
            heatmap_trail = []

            frame_delay = 1.0 / fps

            frame_idx = 0
            while True:
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in lunge_set:
                    current_lunges += 1

                p_pos = player_positions_m[frame_idx]
                if p_pos is not None:
                    heatmap_trail.append(p_pos)
                    if len(heatmap_trail) > 120:
                        heatmap_trail.pop(0)

                kps = player_keypoints_history[frame_idx]
                knee_angles = player_tracker.get_knee_angles(kps)
                frame = drawers.draw_player_pose(frame, kps, knee_angles)
                frame = drawers.draw_ball_trajectory(frame, ball_centers_pixels, frame_idx, tail_len=15)

                b_pos = ball_positions_m[frame_idx]
                frame = drawers.draw_minimap(frame, p_pos, b_pos, heatmap_trail)

                speed = player_speeds_kmh[frame_idx]
                frame = drawers.draw_hud_metrics(frame, speed, current_lunges, t_recovery_rate, frame_idx, fps)

                # Render frame onto Tkinter Canvas
                self.root.after(0, self.render_frame_to_canvas, frame)

                frame_idx += 1
                self.progress["value"] = 50 + (frame_idx / total_frames) * 50

                # Regulate playback speed to real-time FPS
                elapsed = time.time() - t_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            cap.release()
            self.root.after(0, lambda: self.lbl_status.config(text="Status: ✅ Playback & Analysis Complete"))

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Pipeline Error", str(e))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

    def render_frame_to_canvas(self, cv_frame):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 640, 480

        h, w = cv_frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(cv_frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.img_tk = img_tk
        self.canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor="center")

    def update_dashboard_metrics(self, dist, speed, lunges, t_rate):
        self.lbl_val_dist.config(text=f"{dist:.1f} m")
        self.lbl_val_speed.config(text=f"{speed:.1f} km/h")
        self.lbl_val_lunges.config(text=str(lunges))
        self.lbl_val_t_rate.config(text=f"{t_rate:.1f}%")

    def update_coaching_tips(self, t_rate, strikes, distance, lunges):
        self.txt_tips.config(state="normal")
        self.txt_tips.delete("1.0", "end")

        tips = "=== SQUASH AI COACH FEEDBACK ===\n\n"

        tips += "⏱️ T-ZONE RECOVERY:\n"
        if t_rate >= 70:
            tips += "✅ Excellent! You are dominating the T-Zone consistently.\n\n"
        elif t_rate >= 40:
            tips += "⚠️ Moderate. You occasionally stay in corners after hitting. Work on fast side-steps to the T.\n\n"
        else:
            tips += "🔴 Action Needed! You are getting trapped in the corners. Practice ghosting drills back to the center T.\n\n"

        tips += f"🎯 SHOT ANALYSIS:\n- Detected {strikes} ball strikes.\n\n"

        tips += "🦵 BIOMECHANICS & KNEES:\n"
        if lunges > 0:
            tips += f"- Check your knee angles during the {lunges} lunges. Keep your knee angle above 90° for joint safety.\n\n"
        else:
            tips += "- No lunges detected. Try lowering your center of gravity on low corner shots.\n\n"

        self.txt_tips.insert("1.0", tips)
        self.txt_tips.config(state="disabled")

def main():
    root = tk.Tk()
    app = SquashTrainerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
