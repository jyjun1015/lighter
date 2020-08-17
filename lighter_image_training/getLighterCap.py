import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# 파이카메라 초기화
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))
cap = 0

# Load Yolo
net = cv2.dnn.readNet("custom-train-yolo_1000.weights", "custom1/custom-train-yolo.cfg")
classes = ["lighter"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True) :
    img = frame.array
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # detect 결과
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 신뢰도가 0.5 보다 크면 lighter로 카운트
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 카운트된 라이터가 5개 이상이면 캡쳐된 이미지 저장
    if len(indexes) >= 5 :
        cv2.imwrite(str(cap)+".png", img)
        cap += 1