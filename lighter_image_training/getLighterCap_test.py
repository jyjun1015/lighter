import cv2
import numpy as np
import picamera
import os
import io
import time
import multiprocessing

def getCapture(cap) :
    with picamera.PiCamera() as camera :
        camera.resolution = (416, 416)
        while True :
            camera.capture("images/"+str(cap)+".jpg")
            cap += 1
            if cap > 30 :
                return

def yolo(cap) :
    cap_lig = 0
    
    net = cv2.dnn.readNet("yolov3-tiny_3000.weights", "yolov3-tiny.cfg")
    os.chdir('images')
    classes = ["lighter"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    prev = time.time()
    
    while True :
        if os.path.isfile(str(cap)+".jpg") :
            img = cv2.imread(str(cap)+".jpg")
            try :
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            # Object detected
                            center_x = int(detection[0] * 416)
                            center_y = int(detection[1] * 416)
                            w = int(detection[2] * 416)
                            h = int(detection[3] * 416)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

                # 인식된 라이터가 다섯개 이상이면 업로드
                if len(indexes) >= 5 :
                    cv2.imwrite("ok"+str(cap_lig)+".jpg", img)
                    cap_lig += 1

                # 처리가 끝난 이미지는 무조건 삭제
                os.remove(str(cap)+".jpg")
                cap += 1
                prev = time.time()

            except Exception as e :
                print(str(e))
            
        else :
            if time.time() - prev > 10 :
                return
            else :
                pass
        
if __name__ == '__main__' :
    cap = 0
    proc1 = multiprocessing.Process(target=getCapture, args=(cap,))
    proc1.start()
    proc2 = multiprocessing.Process(target=yolo, args=(cap,))
    proc2.start()
    
    proc1.join()
    proc2.join()