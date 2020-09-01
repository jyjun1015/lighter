from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import os

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
        
if __name__ == '__main__' :
    cap_lig = 0
    gDrive = GDrive()
    os.chdir('images')
    
    while True :
        if os.path.isfile("ok"+str(cap_lig)+".jpg") :
            gDrive.upload("ok"+str(cap_lig)+".jpg")
            os.remove("ok"+str(cap_lig)+".jpg")
            cap_lig += 1
        else : pass