import cv2
import numpy as np
import os
import time
from csi_camera import CSI_Camera

# 로컬에 화면 캡쳐 이미지를 저장함
camera = CSI_Camera()
camera.create_gstreamer_pipeline(
sensor_id = 0,
sensor_mode = 2,
framerate = 30,
flip_method = 0,
display_height = 720,
display_width = 1280
)
camera.open(camera.gstreamer_pipeline)
camera.start()
cv2.namedWindow("Sticker Solution", cv2.WINDOW_AUTOSIZE)
cap = 0

if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)

try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("Sticker Solution", 0) >= 0:
            _, img = camera.read()
            cv2.imwrite("datas/"+str(cap)+".jpg", img)
            print(cap, "save")
            time.sleep(0.33)
            cv2.imshow("Sticker Solution", img)
            camera.frames_displayed += 1
            cap = cap + 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
