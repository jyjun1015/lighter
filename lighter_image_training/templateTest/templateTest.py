import cv2

def check(coord_ori, coord_result, target) :
    if coord_result <= coord_ori*(1+target) and coord_result >= coord_ori*(1-target) :
        return True
    else :
        return False

origin = cv2.imread("ligher_ori.jpg")
hurt = cv2.imread("ligher_ori.jpg")

coords = [316, 361, 213, 340]

hurt_modi = hurt.copy()
hurt_modi = hurt[coords[2]:coords[3], coords[0]:coords[1]]

result = cv2.matchTemplate(origin, hurt_modi, cv2.TM_SQDIFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
x, y = minLoc
h, w = hurt_modi.shape[0:2]

coords_result = [x, x+w, y, y+h]
print("original coordinates : ", end='')
print(coords)

for i in range(4) :
    print("predicted coordinates :", coords_result[i], end=' ')
    if not check(coords[i], coords_result[i], 0.03) :
        print("This is faulty!")

dst = cv2.rectangle(origin, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
cv2.imshow("dst", origin)
cv2.waitKey(0)
cv2.destroyAllWindows()