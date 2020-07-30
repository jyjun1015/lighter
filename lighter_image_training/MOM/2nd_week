# 라이터 프로젝트 2주차 회의록

## ■ 진행상황 보고

### 실험환경 준비

#### 라이브러리 설치

현재 라즈베리파이4는 베타버전 (aarch64) OS를 사용하고 있기 때문에 설치과정이 까다로웠습니다. 

1. **OpenCV4** 

    **사용목적: 이미지 또는 영상에 대해 다양한 처리(필터 알고리즘 적용, 편집, 실시간 촬영) **

    + 컴퓨터 비전 관련 오픈소스 라이브러리중 하나로 실시간 이미지 프로세싱을 지원하는 라이브러리입니다.
    + **TensorFlow, Torch(PyTorch), Caffe** 등 딥러닝 프레임워크를 지원합니다.
    + C11, C++14, C#, Java 그리고 Python3을 지원하며 본 프로젝트에서는 Python3을 사용해 개발합니다.

2. **Yolo(You Look Only Once)**

    **사용목적: 화면 속 존재하는 다양한 물체들간 구별 및 식별**

    + 머신러닝 기반의 이미지/영상 속 물체 검출 오픈소스 라이브러리입니다.
    + 이미 학습된 대량의 데이터를 기반으로 화면 속 물체를 인식, 판별 후 식별합니다.
    + 현재 라이브러리에는 ‘가스 라이터’에 대한 학습데이터는 존재하지 않습니다.
        + `PyTorch`를 사용해서 우리가 원하는 `Custom` 데이터셋을 생성해서 Yolo가 라이터를 식별할 수 있게끔 학습시킬 것입니다.
        + 학습 도구는 개인 노트북 또는 필동관 컴퓨터를 사용할 계획입니다.
    + 추가한 데이터를 기반으로 화면 속에서 **라이터를 식별**할 수 있게 됩니다. 

#### 학습데이터 생성

1. **실험변수 설정**

    학습데이터를 생성하기 위해 다양한 실험변수를 설정했습니다.

    | 데이터 실험변수 설정 |         |           |         |           |        |        |
    | :------------------: | ------- | --------- | ------- | --------- | ------ | ------ |
    |        배경색        | 하얀색  | 검정색    | 회색    | 배경 X    |        |        |
    |      배경 조명       | 사용 함 | 사용 안함 |         |           |        |        |
    |   촬영 이미지 크기   | 1080P급 | 720P급    | 540P급  | 360P급    | 240P급 | 120P급 |
    |     라이터 색상      | 빨강색  | 파랑색    | 노랑색  | 보라색    |        |        |
    |  그레이스케일 적용   | 적용 함 | 적용 안함 |         |           |        |        |
    |      필터 적용       | Canny   | Histogram | Sobel   | Laplacian |        |        |
    |      세부 조정       | 밝기    | Sharpness | Erosion | Dilation  |        |        |
    |         거리         | -       |           |         |           |        |        |

2. **실험변수 별 학습데이터 생성 코드 작성 완료**

    *e.g)  이미지 촬영 후 그레이스케일 적용, 세부 조정 그리고 각 필터 적용 후 데이터 저장*

    ```python
    def applyGrayScale(img) :
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def applyCanny(img) :
        return cv2.Canny(img, 50, 200)
    
    def applySobel(img) :
        img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
        img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    
        return cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    
    def applyLaplacian(img) :
        return cv2.Laplacian(img, cv2.CV_32F)
    
    def applyFilter(img, flag) : 
        #grayscale 적용 되어 있어야 함
        if flag[0] :
            #apply histogram
            img = cv2.equalizeHist(img)
        if flag[1] :
            #apply sharpening_1
            kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel_sharpen_1)
        if flag[2] :
            #apply sharpening_2
            kernel_sharpen_2 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
            img = cv2.filter2D(img, -1, kernel_sharpen_2)
        if flag[3] :
            #apply dilation
            kernel_dialation = np.ones((3, 3), np.uint8)
            img = cv2.dilate(img, kernel_dialation, iterations=1)
        return img
    
    for file in os.listdir(image_dir):
        if (file.find('.png') is not -1):       
            src.append(image_dir + file)
            name.append(file)
    
    for i in range(len(src)) :
        filename = name[i].split('.')
        img = cv2.imread(src[i], cv2.IMREAD_COLOR)
        img = applyGrayScale(img)
    
        for j in range(16) :
            temp = j
            flag = [temp/=8, temp/=4, temp/=2, temp/=1]
            img = applyFilter(img, flag)
            cv2.imwrite(filename[0]+'_'+str(j)+'.jpg', img)
    
            img_canny = applyCanny(img)
            cv2.imwrite(filename[0]+'_'+str(j)+'canny'+'.jpg', img_canny)
    
            img_sobel = applySobel(img)
            cv2.imwrite(filename[0]+'_'+str(j)+'sobel'+'.jpg', img_sobel)
    
            img_laplacian = applyLaplacian(img)
            cv2.imwrite(filename[0]+'_'+str(j)+'laplacian'+'.jpg', img_laplacian)
    ```

3. **학습하기 위한 초안 코드 작성 완료 **

    ![image-20200730150546850](C:\Users\cydph\AppData\Roaming\Typora\typora-user-images\image-20200730150546850.png)

    + `image_modeling.py` - 생성한 학습 데이터를 딥러닝 라이브러리에 넣기 위한 모델로 만드는 코드
    + `image_training.py` - 생성한 모델을 가지고 클라우드 서비스로 학습을 돌리기 위한 코드
        현재는 하나의 사물에 대한`custom yolo` 데이터를 만들어야 하므로 가벼운 딥러사이킷런(sklearn)을
    + `image_predict.py` - 학습을 돌리는 코드

## ■ 다음주 진행상황 보고

+ **본격적인 학습 데이터 생성**

    <img src="C:\Users\cydph\AppData\Roaming\Typora\typora-user-images\image-20200730152837095.png" alt="image-20200730152837095" style="zoom:50%;" />

+ **다양한 상황에서도 라이터를 올바르게 인식할 수 있는 `custom dataset `생성**

## ■ 피드백

1. 터널 아이디어
    + 지나가는 통로에 터널을 만들어서 외부 환경 영향으로부터 자유로울 수 있도록 한다.
2. 현장에서 사진 촬영 후 저장 문제
    1. 외부 저장공간으로 처리?
    2. WiFi 연결 후 소켓으로 사진 넘기는 방안 (by 효정)

3. LED 라이트 판넬 
    1. **파이카메라밖에 없을까? 더 좋은건 없을까?**
        + 파이카메라로 일단 많이 찍어봐라. 만일 개선이 필요하다면 꼭 말씀드리기.
    2. **빛반사/역광 등을 고려해야한다.** 
        + 파이카메라로 일단 많이 찍어봐라. 만일 개선이 필요하다면 꼭 말씀드리기.
4. 회의시간에 단순 보고하지말고 우리가 회의시간에 정할 수 있는건 같이 의논하고 정해보자. (30분)
5. 고장 상황에 대한 대처는 반드시 염두하자. -> 백업 (SW든 HW든)
6. 라이터 색상에 따라 가장 평균적으로 높은 정확도 결과를 내는 설정값을 찾아라.
7. 눈으로 보이는 결과를 만들고 감동시켜서 장학금 얻자 (돈)
8. 소규모 공장을 위한 솔루션으로 발전? (우리걸로 만들기 위한 방안 생각하기)
9. **분업**에 대한 고민을 해보자. 
10. 움직이는 물체에 대한 고민도 해야한다.  (적절한 프레임을 찾아야한다.)

---

### 목표

- 다음 주 목요일까지 데이터셋 생성 후 라이터를 식별할 수 있어야 한다. 
- 최대 수요일까지는 사진을 찍어놔야하고, 수요일부터는 학습을 돌려야 한다.

### 지금 당장 할 수 있는 일

+ **없는 것**
    + LED 패널 - 휴대용 후레시 라이트 + 반투명 아크릴 판 (150x300 - 1200원)
        + http://www.slrshop.co.kr/shop/shopdetail.html?branduid=100744&search=PLZ-LED&sort=order&xcode=030&mcode=004&scode=&GfDT=Z293UQ%3D%3D
    + 색지 - 다음 필동관 들릴 때 꼭 사오기
    + 오류가 난 라이터 사진
+ **염두할 사항**
    + LED 패널을 배경으로 사용할지, 카메라 옆에 둬서 물체를 조사하는데에 사용할지

+ **일요일**에 만나서 사진촬영 진행하기
