from ultralytics import YOLO
import numpy as np
import torch

class SquashBallKalmanFilter:
    """
    A simple 2D constant velocity Kalman Filter for tracking squash ball position.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=1.5):
        self.dt = dt
        
        # State: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # State transition matrix (F)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrix (P)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        # Process noise covariance (Q)
        self.Q = np.array([
            [0.25 * (dt**4), 0, 0.5 * (dt**3), 0],
            [0, 0.25 * (dt**4), 0, 0.5 * (dt**3)],
            [0.5 * (dt**3), 0, dt**2, 0],
            [0, 0.5 * (dt**3), 0, dt**2]
        ], dtype=np.float32) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        self.initialized = False

    def init_state(self, x, y):
        self.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self):
        """Predicts the next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, x_meas, y_meas):
        """Updates the state with a new measurement."""
        if not self.initialized:
            self.init_state(x_meas, y_meas)
            return x_meas, y_meas
            
        z = np.array([[x_meas], [y_meas]], dtype=np.float32)
        
        # Innovation (y)
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance (S)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain (K)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state
        self.x = self.x + np.dot(K, y)
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x[0, 0], self.x[1, 0]


class SquashBallTracker:
    """
    Detects and tracks the squash ball using YOLO and a Kalman Filter
    to handle high-speed movements and occlusions.
    """
    def __init__(self, model_path="yolov8n.pt", fps=30.0):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Time step based on video fps for Kalman filter
        dt = 1.0 / float(fps) if fps > 0 else 1.0
        self.kf = SquashBallKalmanFilter(dt=dt)
        self.last_predicted_pos = None



    def detect_frame(self, frame, conf=0.15):
        """
        Runs YOLO to detect the ball.
        Returns:
            bbox: [x1, y1, x2, y2] of the ball, or None.
            conf: confidence score, or None.
        """
        # Run inference
        # Squash ball is small, so we use a lower confidence threshold for detection
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
        result = results[0]
        
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return None, None
            
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        
        # Find index for ball class.
        # In COCO, 'sports ball' is class 32.
        # In custom model, it might be class 0, 1, or 2.
        # Let's search class names containing 'ball'
        ball_idx = -1
        max_ball_conf = 0.0
        
        for idx, (cls_id, c) in enumerate(zip(cls_ids, confs)):
            cls_name = result.names[int(cls_id)].lower()
            if "ball" in cls_name:
                if c > max_ball_conf:
                    max_ball_conf = c
                    ball_idx = idx
                    
        # Fallback to class index 32 if COCO default names are used and name lookup is not matched
        if ball_idx == -1:
            for idx, (cls_id, c) in enumerate(zip(cls_ids, confs)):
                if int(cls_id) == 32: # COCO sports ball
                    if c > max_ball_conf:
                        max_ball_conf = c
                        ball_idx = idx

        if ball_idx != -1:
            return boxes[ball_idx].tolist()[:4], float(confs[ball_idx])
            
        return None, None

    def track_frame(self, frame, conf=0.15):
        """
        Runs YOLO and filters/predicts the ball position using Kalman Filter.
        Returns:
            ball_pos: center point (x, y) or None.
            bbox: bounding box [x1, y1, x2, y2] or None.
            is_predicted: bool indicating if the position was predicted (ball not detected).
        """
        bbox, score = self.detect_frame(frame, conf=conf)
        
        # Kalman Predict
        pred_x, pred_y = self.kf.predict()
        
        if bbox is not None:
            # We detected the ball!
            # Compute center of detected bounding box
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # If the filter was not initialized, init it.
            if not self.kf.initialized:
                self.kf.init_state(cx, cy)
                return (cx, cy), bbox, False
                
            # If the detection is unreasonably far from our prediction,
            # it might be a false positive (unless it is just starting up or after a reset).
            # Max expected movement in 1 frame (e.g. 200 pixels)
            dist_to_pred = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            if dist_to_pred > 250 and self.last_predicted_pos is not None:
                # Treat as occlusion/false positive, fall back to Kalman prediction
                self.last_predicted_pos = (pred_x, pred_y)
                # Predict a bounding box of average size centered around prediction
                w, h = 15, 15
                predicted_bbox = [pred_x - w/2, pred_y - h/2, pred_x + w/2, pred_y + h/2]
                return (pred_x, pred_y), predicted_bbox, True
            
            # Otherwise, update the filter with the new measurement
            ux, uy = self.kf.update(cx, cy)
            self.last_predicted_pos = (ux, uy)
            return (ux, uy), bbox, False
        else:
            # Ball NOT detected: return Kalman prediction
            if self.kf.initialized:
                self.last_predicted_pos = (pred_x, pred_y)
                w, h = 15, 15
                predicted_bbox = [pred_x - w/2, pred_y - h/2, pred_x + w/2, pred_y + h/2]
                return (pred_x, pred_y), predicted_bbox, True
            return None, None, True

    def reset(self):
        """Resets the Kalman Filter."""
        self.kf = SquashBallKalmanFilter(dt=1.0)
        self.last_predicted_pos = None
