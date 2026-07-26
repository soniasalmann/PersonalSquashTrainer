import numpy as np

class SquashAnalytics:
    """
    Computes analytics for a single-player squash practice session:
    - Distance covered (meters)
    - Real-time speed (km/h)
    - Lunges detected
    - T-recovery rate (%)
    """
    def __init__(self, fps=30.0, court_calibrator=None, scale=1.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.calibrator = court_calibrator
        # If frames were scaled during Pass 1, pixel coordinates are in scaled space.
        # Dividing by scale converts them back to original-resolution pixel space
        # so the homography (fitted on the original frame) maps them correctly.
        self.scale = scale if scale > 0 else 1.0
        self.lunge_debounce_frames = int(fps * 1.2) # 1.2 seconds between lunges

    def calculate_player_movement(self, feet_coords_pixel):
        """
        Calculates player positions, distances, and speeds.
        feet_coords_pixel: list of (x, y) coordinates in pixels per frame.
        Returns:
            positions_m: list of (x, y) coordinates in meters.
            speeds_kmh: list of speeds per frame in km/h.
            total_distance: total distance traveled in meters.
        """
        positions_m = []
        speeds_kmh = []
        total_distance = 0.0
        
        # 1. Transform pixels to meters
        for coord in feet_coords_pixel:
            if coord is not None and self.calibrator is not None:
                # Convert scaled pixel coords back to original resolution before applying homography
                orig_x = coord[0] / self.scale
                orig_y = coord[1] / self.scale
                pos = self.calibrator.pixel_to_court(orig_x, orig_y)
                positions_m.append(pos)
            else:
                positions_m.append(None)
                
        # 2. Calculate frame-by-frame speed and accumulate distance
        raw_speeds = []
        for i in range(len(positions_m)):
            if i == 0 or positions_m[i] is None or positions_m[i-1] is None:
                raw_speeds.append(0.0)
                continue
                
            x1, y1 = positions_m[i-1]
            x2, y2 = positions_m[i]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter out tracking keypoint glitches.
            # Max realistic player speed in squash is ~45 km/h ≈ 12.5 m/s.
            # With frame-skipping, dt = 1/effective_fps, so max valid dist = 12.5 * dt.
            max_valid_dist = 12.5 * self.dt   # metres; anything beyond this is a glitch
            if dist > max_valid_dist:
                dist = 0.0

            # Accumulate distance
            total_distance += dist
            
            # Speed in m/s
            speed_ms = dist / self.dt
            raw_speeds.append(speed_ms)

        # 3. Smooth speeds using a moving average window (5 frames)
        window_size = 5
        for i in range(len(raw_speeds)):
            start = max(0, i - window_size // 2)
            end = min(len(raw_speeds), i + window_size // 2 + 1)
            avg_speed_ms = np.mean(raw_speeds[start:end])
            
            # Convert m/s to km/h
            speed_kmh = avg_speed_ms * 3.6
            speeds_kmh.append(float(speed_kmh))
            
        return positions_m, speeds_kmh, float(total_distance)

    def detect_lunges(self, keypoints_list):
        """
        Detects lunges in a sequence of keypoint data.
        A lunge is detected when the vertical hip-to-ankle distance drops significantly
        relative to the torso length (shoulder-to-hip distance).
        Returns:
            lunge_frames: list of frame indices where lunges were detected.
        """
        lunge_frames = []
        last_lunge_frame = -self.lunge_debounce_frames
        
        for i, kps in enumerate(keypoints_list):
            if kps is None:
                continue
                
            conf_thresh = 0.3
            # Use left or right side depending on which is visible
            # Left: Shoulder (5), Hip (11), Ankle (15)
            # Right: Shoulder (6), Hip (12), Ankle (16)
            
            ratios = []
            
            # Left side check
            if (kps[5][2] > conf_thresh and kps[11][2] > conf_thresh and kps[15][2] > conf_thresh):
                torso_len = np.abs(kps[11][1] - kps[5][1])
                hip_ankle_h = np.abs(kps[15][1] - kps[11][1])
                if torso_len > 10:
                    ratios.append(hip_ankle_h / torso_len)
                    
            # Right side check
            if (kps[6][2] > conf_thresh and kps[12][2] > conf_thresh and kps[16][2] > conf_thresh):
                torso_len = np.abs(kps[12][1] - kps[6][1])
                hip_ankle_h = np.abs(kps[16][1] - kps[12][1])
                if torso_len > 10:
                    ratios.append(hip_ankle_h / torso_len)
            
            if len(ratios) == 0:
                continue
                
            # Average ratio across visible sides
            avg_ratio = np.mean(ratios)
            
            # Normal standing ratio is ~1.8 - 2.2. A deep lunge drops this below ~1.35.
            if avg_ratio < 1.30:
                if i - last_lunge_frame >= self.lunge_debounce_frames:
                    lunge_frames.append(i)
                    last_lunge_frame = i
                    
        return lunge_frames

    def analyze_t_recovery(self, player_positions, ball_positions, player_speeds_kmh=None):
        """
        Analyzes T-recovery by identifying shot strike frames, then checking if the player
        returns to the T-zone before the next shot.

        Shot detection uses a 3-tier approach (most reliable first):
          1. Player speed local maxima: peaks in movement speed indicate a lunge/shot moment.
          2. Ball direction reversals: if ball tracking is reliable enough.
          3. Fallback to evenly-spaced estimates if all else fails.

        player_positions:   list of (x, y) court coordinates in meters, per frame.
        ball_positions:     list of (x, y) pixel coordinates (or None), per frame.
        player_speeds_kmh:  list of speeds in km/h per frame (optional but strongly recommended).
        
        Returns:
            strikes: list of frame indices where a shot was hit.
            recovery_rate: percentage of strikes with successful T-recovery.
            recovery_details: list of booleans indicating recovery success per strike.
        """
        if self.calibrator is None:
            return [], 0.0, []

        court_w = float(self.calibrator.dst_points[1][0])
        court_l = float(self.calibrator.dst_points[2][1])
        margin = 1.5  # meters outside court to still accept ball position

        # --- Step 1: Convert valid ball pixel positions to court meters ---
        ball_m = []  # (frame_index, x_m, y_m)
        for i, b_px in enumerate(ball_positions):
            if b_px is None:
                continue
            b_pos = self.calibrator.pixel_to_court(b_px[0], b_px[1])
            if b_pos is None:
                continue
            bx, by = b_pos
            if bx < -margin or bx > court_w + margin or by < -margin or by > court_l + margin:
                continue
            ball_m.append((i, bx, by))

        strikes = []
        strike_debounce = int(self.fps * 0.7)  # min 0.7s between shots

        # --- Tier 1 (Primary): Player speed local maxima ---
        # A lunge to hit the ball creates a local peak in the player's speed.
        # This is reliable regardless of ball detection quality.
        if player_speeds_kmh is not None and len(player_speeds_kmh) > 4:
            speeds = player_speeds_kmh
            # Smooth the speed signal slightly (3-frame window) to reduce noise
            smoothed = []
            for i in range(len(speeds)):
                s = max(0, i - 1)
                e = min(len(speeds), i + 2)
                smoothed.append(float(np.mean(speeds[s:e])))

            # Find local maxima above a minimum speed threshold (5 km/h = ~1.4 m/s)
            min_speed_thresh = 5.0  # km/h
            last_strike = -strike_debounce
            for i in range(1, len(smoothed) - 1):
                if (smoothed[i] > min_speed_thresh and
                        smoothed[i] >= smoothed[i - 1] and
                        smoothed[i] >= smoothed[i + 1] and
                        i - last_strike >= strike_debounce):
                    strikes.append(i)
                    last_strike = i

        # --- Tier 2: Ball direction reversals (supplement if ball data is rich enough) ---
        if len(ball_m) >= 5:
            ball_strikes = []
            last_strike_b = -strike_debounce
            for k in range(1, len(ball_m) - 1):
                fi_prev, x0, y0 = ball_m[k - 1]
                fi_curr, x1, y1 = ball_m[k]
                fi_next, x2, y2 = ball_m[k + 1]
                if (fi_curr - fi_prev) > 10 or (fi_next - fi_curr) > 10:
                    continue
                vx1, vy1 = x1 - x0, y1 - y0
                vx2, vy2 = x2 - x1, y2 - y1
                mag1 = np.sqrt(vx1**2 + vy1**2)
                mag2 = np.sqrt(vx2**2 + vy2**2)
                if mag1 < 0.05 or mag2 < 0.05:
                    continue
                dot = vx1 * vx2 + vy1 * vy2
                if dot < 0 and fi_curr - last_strike_b >= strike_debounce:
                    ball_strikes.append(fi_curr)
                    last_strike_b = fi_curr
            # Merge ball strikes with speed-peak strikes (avoid duplicates within 1s)
            for bf in ball_strikes:
                if not any(abs(bf - sf) < strike_debounce for sf in strikes):
                    strikes.append(bf)
            strikes.sort()

        # --- Tier 3 Fallback: if speed data unavailable, use player position variance ---
        if len(strikes) == 0 and len(player_positions) > 2:
            accel_debounce = int(self.fps * 1.0)
            last_acc = -accel_debounce
            for i in range(2, len(player_positions)):
                p0 = player_positions[i - 2]
                p1 = player_positions[i - 1]
                p2 = player_positions[i]
                if p0 is None or p1 is None or p2 is None:
                    continue
                v1 = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) * self.fps
                v2 = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * self.fps
                if v2 > 1.5 and v1 < 0.5 and i - last_acc >= accel_debounce:
                    strikes.append(i)
                    last_acc = i

        if not strikes:
            return [], 0.0, []

        # --- Check T-recovery for each strike ---
        recovery_details = []
        for idx, strike_f in enumerate(strikes):
            # Window: from this strike to next strike (or up to 3 seconds after last shot)
            start_f = strike_f + 1
            end_f = strikes[idx + 1] if idx + 1 < len(strikes) else min(len(player_positions), strike_f + int(self.fps * 3.0))

            recovered = False
            for f in range(start_f, end_f):
                p_pos = player_positions[f]
                if p_pos is not None:
                    if self.calibrator.is_in_t_zone(p_pos[0], p_pos[1]):
                        recovered = True
                        break
            recovery_details.append(recovered)

        recovery_rate = (sum(recovery_details) / len(recovery_details)) * 100.0 if recovery_details else 0.0
        return strikes, float(recovery_rate), recovery_details
