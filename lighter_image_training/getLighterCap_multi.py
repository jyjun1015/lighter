import cv2
import numpy as np
import picamera
import io
import os
import time
import multiprocessing

def getCapture(cap) :
    with picamera.PiCamera() as camera :
        camera.resolution = (416, 416)
        stream = io.BytesIO()

        
        for frame in camera.capture_continuous(stream, format="jpeg") :
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, 1)
            cv2.imwrite("images/"+str(cap)+".png", img)
            cap += 1

def yolo(cap) :
    net = cv2.dnn.readNet("custom-train-yolo_2000.weights", "custom-train-yolo.cfg")
    classes = ["lighter"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    while True :
        if os.path.isfile("images/"+str(cap)+".png") :
            img = cv2.imread("images/"+str(cap)+".png")
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
                if len(indexes) < 5 :
                    os.remove("images/"+str(cap)+".png")
                print("yes")
                cap += 1
            except Exception as e :
                print(str(e))
            
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