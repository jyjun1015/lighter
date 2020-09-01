import cv2
import numpy as np
import picamera
import io
import os
import time
import multiprocessing
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

class GDrive :
    def __init__(self):
        try :
            import argparse
            flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
        except ImportError:
            flags = None

        # 현재는 py파일과 client_secret_drive.json 파일이 모두 사진폴더에 있어야 함.
        # 이후 간단히 '.jpg' 확장자만 업로드할 수 있도록 바꿔주면 된다. 
        # os.chdir('images')       # 현재 디렉토리를 사진폴더로 변경
        SCOPES = 'https://www.googleapis.com/auth/drive.file'   # [변경X] Google API 사용 로그인 링크
        store = file.Storage('storage.json')                    # 현재 로그인된 드라이브 정보 로드.
        creds = store.get()                                     # storage.json 파일 가져온다.

        if not creds or creds.invalid:              # 만일 storage.json 파일이 없다면 생성해야 함.
            print("make new storage data file ")    # 구글 로그인을 시도하고 storage.json 파일 생성.
            flow = client.flow_from_clientsecrets('client_secret_drive.json', SCOPES)
            creds = tools.run_flow(flow, store, flags) \
                    if flags else tools.run(flow, store)

        self.DRIVE = build('drive', 'v3', http=creds.authorize(Http()))
        self.FOLDER = '1k0y5cctckms5m8F9UDg3mqrrkgIbg2Ax'        # 저장할 드라이브 폴더 고유 ID
    
    # 이전엔 폴더의 기존 이미지를 전부 업로드 했다면 이제는 파일을 하나로 특정하여 업로드
    def upload(self, filename) :
        metadata = {'name': filename,
                    'parents':[self.FOLDER],                 # 사진 저장할 드라이브 내 폴더
                    'mimeType': None                    # 데이터 타입은 아랫줄에서 결정.
                    }

        res = self.DRIVE.files().create(body=metadata, media_body=filename).execute()

def getCapture(cap) :
    with picamera.PiCamera() as camera :
        camera.resolution = (416, 416)
        while True :
            camera.capture("images/"+str(cap)+".jpg")
            cap += 1
            if cap > 30 :
                return
        """stream = io.BytesIO()
        
        for frame in camera.capture_continuous(stream, format="jpeg") :
            data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, 1)
            cv2.imwrite("images/"+str(cap)+".jpg", img)
            cap += 1

            if cap > 30 :
                return"""

def yolo(cap) :
    gDrive = GDrive()

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
                    gDrive.upload(str(cap)+".jpg")

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