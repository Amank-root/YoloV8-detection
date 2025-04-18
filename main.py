from ultralytics import YOLO
import cv2
import numpy as np

# --- CONFIG ---
model_path = "model/best.pt" # change this to your model path
video_path = "assets/conv.mp4" # change this to your camera for live detection
# video_path = 0 # uncomment this for live detection
output_path = "detected_bars_only_with_stop_alert.mp4" # change this to your output video path
# ----------------

# Load YOLOv8 model
model = YOLO(model_path) # Load the trained model
# model.fuse() # Fuse model layers for faster inference

# Video input/output
cap = cv2.VideoCapture(video_path) # change this to your camera for live detection
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # comment this when in production
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # comment this when in production
fps = cap.get(cv2.CAP_PROP_FPS) # comment this when in production
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # comment this when in production
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # comment this when in production

# Movement tracking
prev_centroids = [] 
still_frame_count = 0 
belt_stopped = False 
stop_threshold = int(fps * 1)  # ~1.0 seconds
movement_tolerance = 2.0  # pixel threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    # Detect only bars (class 0)
    centroids = []
    for box, cls in zip(boxes, classes):
        if cls == 0:  # Only consider bars
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centroids.append((cx, cy))

    if prev_centroids and len(centroids) == len(prev_centroids):
        movement = [np.linalg.norm(np.array(c1) - np.array(c2))
                    for c1, c2 in zip(centroids, prev_centroids)]
        avg_movement = np.mean(movement)

        if avg_movement < movement_tolerance:
            still_frame_count += 1
        else:
            still_frame_count = 0
    else:
        still_frame_count = 0

    prev_centroids = centroids

    if still_frame_count >= stop_threshold:
        belt_stopped = True
        # print("Conveyor Halted")
    else:
        belt_stopped = False
        # print("belt running")

    frame_annotated = frame.copy()
    for box, cls in zip(boxes, classes):
        if cls == 0:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_annotated, "Bar", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if belt_stopped:
        cv2.putText(frame_annotated, "Conveyor Halted", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
    else:
        cv2.putText(frame_annotated, "OK", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)

    out.write(frame_annotated) # comment this when in production
    # cv2_imshow(frame_annotated) # uncomment this if want to see annotated frames in colab
    cv2.imshow("Bar Detection Using YoloV8", frame_annotated) # uncomment this if running locally
    if cv2.waitKey(1) & 0xFF == ord('q'): # uncomment this if running locally
        break

cap.release()
out.release() # comment this when in production
cv2.destroyAllWindows() # uncomment in production


print("✅ Saved video with only bar detection and belt stop alert:", output_path)
