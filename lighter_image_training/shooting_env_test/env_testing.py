import os
import cv2
import numpy as np

src = []
name = []
image_dir = "C:\\Users\\hyoj\\OneDrive\\Desktop\\lighter\\main\\lighter_image_training\\shooting_env_test\\image_sample\\"

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