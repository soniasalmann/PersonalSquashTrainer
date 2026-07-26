import cv2
import numpy as np

class SquashDrawers:
    """
    Renders annotations, pose skeleton, ball trajectory, speed metrics,
    and a top-down minimap on video frames.
    """
    # Keypoint connections for skeleton
    SKELETON_CONNECTIONS = [
        (5, 7), (7, 9),        # left arm: L-shoulder->L-elbow->L-wrist
        (6, 8), (8, 10),       # right arm: R-shoulder->R-elbow->R-wrist
        (11, 13), (13, 15),    # left leg: L-hip->L-knee->L-ankle
        (12, 14), (14, 16),    # right leg: R-hip->R-knee->R-ankle
        (5, 6), (11, 12),      # shoulders, hips
        (5, 11), (6, 12)       # torso diagonals
    ]

    def __init__(self, primary_color=(0, 255, 0), ball_color=(0, 255, 255), 
                 skeleton_color=(255, 180, 50), text_color=(255, 255, 255)):
        self.primary_color = primary_color
        self.ball_color = ball_color
        self.skeleton_color = skeleton_color
        self.text_color = text_color

    def draw_player_pose(self, frame, keypoints, knee_angles=None):
        """Draws skeleton and joint keypoints."""
        if keypoints is None:
            return frame

        h, w = frame.shape[:2]
        conf_thresh = 0.3

        # Draw skeleton lines
        for kp1_idx, kp2_idx in self.SKELETON_CONNECTIONS:
            kp1 = keypoints[kp1_idx]
            kp2 = keypoints[kp2_idx]
            if kp1[2] > conf_thresh and kp2[2] > conf_thresh:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(frame, pt1, pt2, self.skeleton_color, 2, cv2.LINE_AA)

        # Draw joint points
        for i, kp in enumerate(keypoints):
            if kp[2] > conf_thresh:
                pt = (int(kp[0]), int(kp[1]))
                # Color code: head points = blue, body = red/orange
                color = (255, 0, 0) if i < 5 else (0, 0, 255)
                cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

        # Draw knee angles if available
        left_angle, right_angle = knee_angles if knee_angles else (None, None)
        
        # Left knee is joint 13
        if left_angle is not None and keypoints[13][2] > conf_thresh:
            pt = (int(keypoints[13][0]) - 10, int(keypoints[13][1]))
            cv2.putText(frame, f"{int(left_angle)}*", pt, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        
        # Right knee is joint 14
        if right_angle is not None and keypoints[14][2] > conf_thresh:
            pt = (int(keypoints[14][0]) + 10, int(keypoints[14][1]))
            cv2.putText(frame, f"{int(right_angle)}*", pt, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_ball_trajectory(self, frame, trajectory_points, current_idx, tail_len=15):
        """Draws a fading trajectory line for the ball."""
        start = max(0, current_idx - tail_len)
        for i in range(start, current_idx):
            pt1 = trajectory_points[i]
            pt2 = trajectory_points[i + 1]
            if pt1 is None or pt2 is None:
                continue
                
            # Compute fading alpha
            alpha = (i - start) / (current_idx - start)
            thickness = int(1 + 4 * alpha)
            color = (
                int(self.ball_color[0] * alpha),
                int(self.ball_color[1] * alpha),
                int(self.ball_color[2] * alpha)
            )
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 
                     color, thickness, cv2.LINE_AA)
            
        # Draw current ball center
        current_pt = trajectory_points[current_idx]
        if current_pt is not None:
            cv2.circle(frame, (int(current_pt[0]), int(current_pt[1])), 6, self.ball_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(current_pt[0]), int(current_pt[1])), 7, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_minimap(self, frame, player_pos_m, ball_pos_m, heatmap_points, width=150, height=228, margin=20, court_w=6.4, court_l=9.75):
        """
        Draws a top-down minimap of the squash court showing player and ball positions.
        """
        f_h, f_w = frame.shape[:2]
        
        # Minimap ROI bounds (Top Right Corner)
        x_start = f_w - width - margin
        y_start = margin
        x_end = f_w - margin
        y_end = margin + height
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (30, 30, 30), -1)
        # Apply overlay with alpha=0.85
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw map outline
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (200, 200, 200), 2, cv2.LINE_AA)
        
        # Helper to map meter coordinates to minimap pixel space
        def meter_to_pixel(x_m, y_m):
            # Width = court_w meters, Height = court_l meters
            # Origin is bottom-left (x=0, y=0)
            mx = x_start + int((x_m / court_w) * width)
            my = y_end - int((y_m / court_l) * height)
            return mx, my

        # Draw court lines
        # 1. Short Line: horizontal floor line at 4.26m from back (y=4.26)
        sl_x1, sl_y1 = meter_to_pixel(0.0, 4.26)
        sl_x2, sl_y2 = meter_to_pixel(6.4, 4.26)
        cv2.line(frame, (sl_x1, sl_y1), (sl_x2, sl_y2), (180, 180, 180), 1, cv2.LINE_AA)
        
        # 2. Half-court Line: vertical floor line in back half (x=3.2, y in [0.0, 4.26])
        hc_x1, hc_y1 = meter_to_pixel(3.2, 0.0)
        hc_x2, hc_y2 = meter_to_pixel(3.2, 4.26)
        cv2.line(frame, (hc_x1, hc_y1), (hc_x2, hc_y2), (180, 180, 180), 1, cv2.LINE_AA)
        
        # 3. T-Zone center boundary (1.5m radius circle around [3.2, 4.26])
        tx, ty = meter_to_pixel(3.2, 4.26)
        # Calculate pixel radius (1.5m)
        r_pixel = int((1.5 / 6.4) * width)
        cv2.circle(frame, (tx, ty), r_pixel, (100, 100, 100), 1, cv2.LINE_8)
        
        # 4. Service boxes (Red boxes in real squash, we'll draw simple outlines)
        # Left box: x in [0.0, 1.6], y in [4.26, 4.26-1.6]
        # Right box: x in [4.8, 6.4], y in [4.26, 4.26-1.6]
        # (For MVP, short line & half line are sufficient)

        # Draw Player Heatmap / Trail
        for i in range(len(heatmap_points) - 1):
            p1 = heatmap_points[i]
            p2 = heatmap_points[i+1]
            if p1 is not None and p2 is not None:
                px1, py1 = meter_to_pixel(p1[0], p1[1])
                px2, py2 = meter_to_pixel(p2[0], p2[1])
                # Draw trailing green line
                cv2.line(frame, (px1, py1), (px2, py2), (0, 200, 0), 1, cv2.LINE_AA)

        # Draw Player Position (Green dot)
        if player_pos_m is not None:
            px, py = meter_to_pixel(player_pos_m[0], player_pos_m[1])
            if x_start <= px <= x_end and y_start <= py <= y_end:
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Ball Position (Yellow dot)
        if ball_pos_m is not None:
            bx, by = meter_to_pixel(ball_pos_m[0], ball_pos_m[1])
            if x_start <= bx <= x_end and y_start <= by <= y_end:
                cv2.circle(frame, (bx, by), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Label the map
        cv2.putText(frame, "COURT MAP", (x_start + 5, y_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def draw_hud_metrics(self, frame, speed_kmh, lunge_count, t_rate, current_frame, fps):
        """Draws current training metrics (speed, lunges, T-recovery rate) in a HUD box."""
        # Top-left HUD box
        x_start, y_start = 20, 20
        width, height = 240, 110
        
        # Transparent background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + width, y_start + height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (120, 120, 120), 1, cv2.LINE_AA)
        
        # Render text metrics
        cv2.putText(frame, "SQUASH PERSONAL TRAINER", (x_start + 10, y_start + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (x_start + 10, y_start + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"Lunges Completed: {lunge_count}", (x_start + 10, y_start + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"T-Recovery Rate: {t_rate:.1f}%", (x_start + 10, y_start + 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
                    
        # Timer / Time stamp
        total_sec = current_frame / fps
        mins = int(total_sec // 60)
        secs = int(total_sec % 60)
        cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (x_start + 10, y_start + 102), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return frame
