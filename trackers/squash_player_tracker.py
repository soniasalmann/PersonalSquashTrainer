from ultralytics import YOLO
import numpy as np
import torch

class SquashPlayerTracker:
    """
    Tracks a single player's pose, keypoints, and biomechanics (lunges, posture)
    using YOLOv8-pose.
    """
    def __init__(self, model_path="models/yolov8m-pose.pt"):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_frame(self, frame):
        """Runs pose estimation on a single frame."""
        # Run inference
        results = self.model.predict(frame, conf=0.3, device=self.device, verbose=False)
        return results[0]

    def get_player_keypoints(self, result):
        """
        Extracts keypoints for the main player (solo practice).
        If multiple people are detected, chooses the one with the largest bounding box area.
        Returns:
            keypoints: ndarray of shape (17, 3) representing [x, y, conf], or None if no one is detected.
            bbox: list [x1, y1, x2, y2] representing the bounding box, or None.
        """
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return None, None
            
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints_obj = getattr(result, "keypoints", None)
        
        if keypoints_obj is None or getattr(keypoints_obj, "data", None) is None:
            return None, None
            
        kps_data = keypoints_obj.data.cpu().numpy() # shape (N, 17, 3) where N is number of people
        
        if len(boxes) == 0 or len(kps_data) == 0:
            return None, None
            
        # Select the index of the largest bounding box (heuristically the active player on court)
        best_idx = 0
        max_area = 0
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_idx = idx
                
        return kps_data[best_idx], boxes[best_idx].tolist()

    @staticmethod
    def angle_bw_points(a, b, c):
        """Computes the angle ABC (at point B) in degrees."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = np.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = np.sqrt(bc[0]**2 + bc[1]**2)

        if mag_ba == 0 or mag_bc == 0:
            return None

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))

    def get_knee_angles(self, keypoints):
        """
        Calculates left and right knee angles.
        Keypoint indices:
        - Left: Hip (11), Knee (13), Ankle (15)
        - Right: Hip (12), Knee (14), Ankle (16)
        """
        if keypoints is None:
            return None, None

        # Check if coordinates have acceptable confidence
        conf_thresh = 0.3
        
        # Left Knee
        left_hip = keypoints[11]
        left_knee = keypoints[13]
        left_ankle = keypoints[15]
        
        left_angle = None
        if (left_hip[2] > conf_thresh and left_knee[2] > conf_thresh and left_ankle[2] > conf_thresh):
            left_angle = self.angle_bw_points(left_hip[:2], left_knee[:2], left_ankle[:2])
            
        # Right Knee
        right_hip = keypoints[12]
        right_knee = keypoints[14]
        right_ankle = keypoints[16]
        
        right_angle = None
        if (right_hip[2] > conf_thresh and right_knee[2] > conf_thresh and right_ankle[2] > conf_thresh):
            right_angle = self.angle_bw_points(right_hip[:2], right_knee[:2], right_ankle[:2])
            
        return left_angle, right_angle

    def get_player_feet_center(self, keypoints):
        """
        Returns the middle point between the left and right ankle keypoints (in pixels).
        Used to track the player's 2D position on the court floor.
        """
        if keypoints is None:
            return None
            
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        conf_thresh = 0.3
        
        valid_l = left_ankle[2] > conf_thresh
        valid_r = right_ankle[2] > conf_thresh
        
        if valid_l and valid_r:
            cx = (left_ankle[0] + right_ankle[0]) / 2.0
            cy = (left_ankle[1] + right_ankle[1]) / 2.0
            return float(cx), float(cy)
        elif valid_l:
            return float(left_ankle[0]), float(left_ankle[1])
        elif valid_r:
            return float(right_ankle[0]), float(right_ankle[1])
            
        # Fallback to hips if ankles are occluded
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if left_hip[2] > conf_thresh and right_hip[2] > conf_thresh:
            cx = (left_hip[0] + right_hip[0]) / 2.0
            cy = (left_hip[1] + right_hip[1]) / 2.0
            return float(cx), float(cy)
            
        return None
