from ultralytics import YOLO
import cv2
import math
import numpy as np
import cvzone
from sort import *


classnames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_name(inp):
    return inp.split('/')[-1].split('.')[0]

video_path = "./video/video8.mp4"


model_path = "../YOLO-Weights/yolov8x.pt"
if os.path.exists(f"{get_name(model_path)}_{get_name(video_path)}.mp4"):
    output_video_name = f"{get_name(model_path)}_{get_name(video_path)}v2.mp4"
else:
    output_video_name = f"{get_name(model_path)}_{get_name(video_path)}.mp4"
model = YOLO(model_path)

tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))


while True:
    success, image = cap.read()
    if not success:
        break
    # 2 contrast 10 brightness
    image = cv2.addWeighted( image, 2, image, 0, 10)
    results = model(image, stream=True)
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            classname = classnames[int(box.cls)]
            conf = math.ceil(box.conf[0] * 100) / 100

            if classname == "bird":
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResult = tracker.update(detections)
    id_no = 0
    
    for results in trackerResult:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(image, f"{id_no}", (max(0, x1), max(35, y1)), thickness=1, scale=2, offset=3)
        id_no += 1
        cvzone.cornerRect(image, (x1, y1, w, h), l=9, colorR=(255, 0, 0))
    
    # Write the frame to the output video
    # cv2.imshow("Result", image)
    # if cv2.waitKey(1) == 13:
    #     break
    out.write(image)

cv2.destroyAllWindows()
cap.release()
out.release()


