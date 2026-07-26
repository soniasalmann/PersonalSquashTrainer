import cv2
import numpy as np
import json
import os

class CourtCalibrator:
    """
    Handles perspective homography to map 2D video pixel coordinates
    to actual 2D court coordinates in meters.
    
    Standard Squash Court Dimensions:
    - Width: 6.4 meters
    - Length: 9.75 meters
    - T-Zone intersection: (3.2m, 4.26m) from back-left corner (0.0, 0.0)
    """
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.homography_matrix = None
        self.inverse_homography_matrix = None
        self.src_points = None
        
        # Target court coordinates in meters (Origin (0,0) at back-left floor corner)
        # Sequence of dst points: [Back-Left, Back-Right, Front-Left, Front-Right]
        self.dst_points = np.array([
            [0.0, 0.0],       # Back-Left
            [6.4, 0.0],       # Back-Right
            [0.0, 9.75],      # Front-Left
            [6.4, 9.75]       # Front-Right
        ], dtype=np.float32)

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check for custom court length or half-court mode in config
        court_length = config.get("court_length_m", 4.26 if config.get("is_half_court", False) else 9.75)
        court_width = config.get("court_width_m", 6.4)
        
        self.dst_points = np.array([
            [0.0, 0.0],            # Back-Left
            [court_width, 0.0],    # Back-Right
            [0.0, court_length],   # Front-Left
            [court_width, court_length] # Front-Right
        ], dtype=np.float32)

        try:
            self.src_points = np.array([
                config["back_wall_left"],
                config["back_wall_right"],
                config["front_wall_left"],
                config["front_wall_right"]
            ], dtype=np.float32)
            
            # Compute homography matrices
            self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.inverse_homography_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            print(f"[CourtCalibrator] Homography initialized from {config_path} (Court Length: {court_length}m)")
        except KeyError as e:
            print(f"[CourtCalibrator] Error: Missing required key in config: {e}")
            raise

    def set_calibration_points(self, back_left, back_right, front_left, front_right):
        """Manually set source points and calculate homography."""
        self.src_points = np.array([
            back_left,
            back_right,
            front_left,
            front_right
        ], dtype=np.float32)
        self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_homography_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def pixel_to_court(self, px, py):
        """
        Converts pixel coordinates (px, py) in the video frame
        to real-world court coordinates (X, Y) in meters.
        """
        if self.homography_matrix is None:
            return None
            
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        x_m, y_m = transformed[0][0]
        return float(x_m), float(y_m)

    def court_to_pixel(self, x_m, y_m):
        """
        Converts real-world court coordinates (x_m, y_m) in meters
        to pixel coordinates (px, py) in the video frame.
        """
        if self.inverse_homography_matrix is None:
            return None
            
        point = np.array([[[x_m, y_m]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography_matrix)
        px, py = transformed[0][0]
        return int(px), int(py)

    def is_in_t_zone(self, x_m, y_m, radius=2.0):
        """
        Checks if the position (x_m, y_m) in meters is within the 'T' zone.
        T center is at the mid-width and the short-line distance from the back wall.
        For a full court: short line is at 4.26m, so T is at (3.2, 4.26).
        For a half court (4.26m long): short line IS the front wall, so T is at mid-court (~2.13m).
        We dynamically place the T at mid-width and the short-line Y (which is the min of court_length and 4.26m).
        """
        court_w = float(self.dst_points[1][0])  # e.g. 6.4m
        court_l = float(self.dst_points[2][1])  # e.g. 4.26m or 9.75m
        t_x = court_w / 2.0                     # always mid-width
        # Short line is 4.26m from back wall on a full court.
        # On a half court the court IS 4.26m, so T is near mid-court.
        t_y = min(4.26, court_l * 0.5)
        dist = np.sqrt((x_m - t_x)**2 + (y_m - t_y)**2)
        return dist <= radius
