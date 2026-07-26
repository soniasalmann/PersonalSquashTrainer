import cv2
import json
import os

# Order of clicks:
# 1. Back-Left (bottom-left of floor in frame)
# 2. Back-Right (bottom-right of floor in frame)
# 3. Front-Left (top-left of floor in frame, where front wall meets floor)
# 4. Front-Right (top-right of floor in frame, where front wall meets floor)

points = []
video_path = "input_videos/practice.mp4"
config_path = "court_config.json"

def click_event(event, x, y, flags, params):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        # Draw a small circle at the clicked point
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        # Put text indicating point number
        labels = ["Back-Left", "Back-Right", "Front-Left", "Front-Right"]
        label = labels[len(points)-1] if len(points) <= 4 else str(len(points))
        cv2.putText(img_display, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Calibrator", img_display)
        
        if len(points) == 4:
            print("\nAll 4 points selected:")
            print(f"Back-Left: {points[0]}")
            print(f"Back-Right: {points[1]}")
            print(f"Front-Left: {points[2]}")
            print(f"Front-Right: {points[3]}")
            save_config()
            print("Successfully saved to court_config.json! You can now close this window.")

def save_config():
    config_data = {
        "back_wall_left": points[0],
        "back_wall_right": points[1],
        "front_wall_left": points[2],
        "front_wall_right": points[3],
        "notes": "Calibrated using calibrate_court.py"
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

def main():
    global img_display
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at '{video_path}'")
        return
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read the first frame of the video.")
        return
        
    img_display = frame.copy()
    
    print("=== SQUASH COURT CALIBRATOR ===")
    print("Click on the 4 floor corners in this EXACT order:")
    print("1. Bottom-Left corner of the floor (Back-Left)")
    print("2. Bottom-Right corner of the floor (Back-Right)")
    print("3. Top-Left corner of the floor (Front-Left, where floor meets front wall)")
    print("4. Top-Right corner of the floor (Front-Right, where floor meets front wall)")
    print("\nPress 'r' to reset points, or 'q' to quit.")
    
    cv2.namedWindow("Calibrator")
    cv2.setMouseCallback("Calibrator", click_event)
    
    while True:
        cv2.imshow("Calibrator", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            img_display = frame.copy()
            print("\nPoints reset. Start clicking again in order.")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
