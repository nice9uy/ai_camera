import cv2
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
import torch

# # Load the YOLOv8 model
# # Load the YOLOv8 model with trust_repo=True
# model = torch.hub.load('ai_model/yolov8n.pt', 'yolov8n', source='local')

from ultralytics import YOLO
model = YOLO('ai_model/yolov8n.pt')

def gen():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)

        # Draw bounding boxes and labels
        for box in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]}: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')
