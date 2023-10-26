import cv2
import math
import numpy as np
from sort import *
from ultralytics import YOLO
import cvzone


def calculate_initial_speed(frame, frame_count, frame_interval):
    if frame_count <= frame_interval:
        return None

    # Capture the first and second frames
    if frame_count == frame_interval + 1:
        frame1 = frame
    elif frame_count == frame_interval + 2:
        frame2 = frame

        # Calculate the Euclidean distance between points in the two frames
        points1 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), 100, 0.3, 7)
        points2, status, error = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), points1, None)

        if status is not None:
            distance = np.linalg.norm(points2 - points1, axis=2)
            avg_distance = np.mean(distance)
            initial_speed = avg_distance / (frame_interval / 30.0)  # Assuming 30 FPS

            return initial_speed


def calculate_and_display_trajectory(initial_speed, weight_kg, g, canvas_width, canvas_height, scale,num_points):
    # Initialize a blank canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # launch angle (in degrees)
    launch_angle = math.degrees(math.asin((weight_kg * g) / initial_speed))

    # time of flight (2 * V0 * sin(theta) / g)
    time_of_flight = (2 * initial_speed * math.sin(math.radians(launch_angle))) / g

    
    horizontal_distance = initial_speed * math.cos(math.radians(launch_angle)) * time_of_flight

    
    x0, y0 = 0, canvas_height - 50  # The launch point

   
    vx0 = initial_speed * math.cos(math.radians(launch_angle))
    vy0 = initial_speed * math.sin(math.radians(launch_angle))

    
    time_step = 0.1  
    current_time = 0

    # while current_time <= time_of_flight:
    #     # Calculate the current position
    #     x = x0 + vx0 * current_time
    #     y = y0 - (vy0 * current_time - 0.5 * g * current_time ** 2)

    #     # Draw a point on the canvas (scaled)
    #     x_scaled = int(x * scale)
    #     y_scaled = int(y * scale)
    #     cv2.circle(canvas, (x_scaled, y_scaled), 2, (0, 255, 0), -1)

    #     # Update time
    #     current_time += time_step
    for i in range(1, num_points + 1):
        t = i * 0.1 
        x = initial_speed * t
        y = (initial_speed * t - 0.5 * g * t ** 2)
        x, y = int(x * scale), int(y * scale)
        
        cv2.circle(canvas, (x, canvas_height - y), 5, (0, 0, 255), -1)

    # Display the trajectory
    cv2.imshow("Clay Target Trajectory", canvas)
    cv2.waitKey(0)
    


weight_kg = 0.105 
g = 9.81  
frame_interval = 5 

def get_name(inp):
    return inp.split('/')[-1].split('.')[0]

video_path = "./video/skeet6.mp4"
model_path = "../YOLO-Weights/best_skeet.pt"
output_video_name = f"{get_name(model_path)}_predicted_{get_name(video_path)}.mp4"
model = YOLO(model_path)
tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)


cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
frame_no = 0
initial_speed = None
object_tracks = {}  
predictions = {}    
prev_frame = None
optical_flow = None

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_no <= frame_interval:
        initial_speed = calculate_initial_speed(frame, frame_no, frame_interval)
    else:
        if initial_speed is not None:
            calculate_and_display_trajectory(initial_speed, weight_kg, g, 800, 600, 10)
            initial_speed = None 

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        optical_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, optical_flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w, _ = frame.shape
        shift_x = optical_flow[..., 0].mean()
        shift_y = optical_flow[..., 1].mean()
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        frame = cv2.warpAffine(frame, M, (w, h))

    prev_frame = gray
    if frame_no % 15 == 0 and frame_no != 0:
        object_tracks = {}
        predictions = {}  
    results = model(frame, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100

            if conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResult = tracker.update(detections)

    for results in trackerResult:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        if id in object_tracks:
            object_tracks[id].append((x1 + w / 2, y1 + h / 2))
        else:
            object_tracks[id] = [(x1 + w / 2, y1 + h / 2)]

        for id, path in object_tracks.items():
          if len(path) >= 2:
            x_prev, y_prev = path[-2]
            x_curr, y_curr = path[-1]

            # Compensate for camera motion
            x_curr += shift_x
            y_curr += shift_y

            delta_x = x_curr - x_prev
            delta_y = y_curr - y_prev

            # Predict future positions for the next few frames
            num_frames_to_predict = 5
            predicted_path = [(x_curr + delta_x * i, y_curr + delta_y * i) for i in range(1, num_frames_to_predict + 1)]
            predictions[id] = predicted_path

        # Draw the tracks and predictions
        cvzone.putTextRect(frame, f"{id}", (max(0, x1), max(35, y1)), thickness=1, scale=2, offset=3)
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, colorR=(255, 0, 0))

    # Draw predicted paths
    for id, path in predictions.items():
        for i, (x, y) in enumerate(path):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cvzone.putTextRect(frame, f"{i+1}", (int(x), int(y)), thickness=1, scale=1, offset=1, colorB=(0, 0, 255))

            cvzone.putTextRect(frame, f"{frame_no}", (50, 50), thickness=1, scale=2, offset=3)
    frame_no += 1

    # Write the frame to the output video
    out.write(frame)
    # cv2.imshow("Result", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
