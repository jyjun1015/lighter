import cv2
import numpy as np
from csi_camera import CSI_Camera

DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 576
SENSOR_MODE = 2         # 1920x1080, 30fps: 2, 1280x720, 60fps: 3

# Load YOLO
net = cv2.dnn.readNet(
    "yolov3-tiny_4000.weights",
    "yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = ["head", "body"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def draw_label(img, lbl_txt, lbl_pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    cv2.putText(img, lbl_txt, lbl_pos, font_face, scale, color, 1, cv2.LINE_AA)

def read_camera(csi_camera):
    _, img = csi_camera.read()
    draw_label(img, "Displayed FPS: " + str(csi_camera.last_frames_displayed), (10, 20))
    return img

def lighter_detect():
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id       = 0,
        sensor_mode     = SENSOR_MODE,
        framerate       = 30,
        flip_method     = 0,
        display_height  = DISPLAY_HEIGHT,
        display_width   = DISPLAY_WIDTH,
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("Lighter Test FPS", cv2.WINDOW_AUTOSIZE)

    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("Lighter Test FPS", 0) >= 0:
            temp = read_camera(camera)
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

            height, width = temp.shape
            img = np.zeros([height, width, 3])

            for i in range(3) :
            	img[:,:,i] = temp
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img = cv2.convertScaleAbs(img)
            blob = cv2.dnn.blobFromImage(img, 0.00392, (446, 446), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 :
                        center_x = int(detection[0] * DISPLAY_WIDTH)
                        center_y = int(detection[1] * DISPLAY_HEIGHT)
                        w = int(detection[2] * DISPLAY_WIDTH)
                        h = int(detection[3] * DISPLAY_HEIGHT)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        if class_id == 0: cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)
                        elif class_id == 1: cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2, cv2.LINE_AA)
                        else: cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, classes[class_id], (x, y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            cv2.imshow("Lighter Test FPS", img)
            camera.frames_displayed += 1
            if (cv2.waitKey(5) & 0xFF) == 27: break # ESC key Stops program
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lighter_detect()

