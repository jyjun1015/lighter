import cv2

def check(coord_ori, coord_result, target) :
    if coord_result <= coord_ori*(1+target) and coord_result >= coord_ori*(1-target) :
        return True
    else :
        return False

def resize(file) :
    target = cv2.imread(file)
    target = cv2.resize(target, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return target

origin = resize("origin3.jpg")
dst = origin.copy()
hurt = resize("fault4.jpg")
dst2 = hurt.copy()

#coords_origin = [[363, 452, 352, 594], [465, 544, 352, 596], 
#        [559, 640, 357, 596], [658, 726, 358, 595], [750, 819, 358, 594]]

#coords_origin = [[658, 735, 404, 657]]

coords_origin = [[226, 290, 311, 560], [311, 373, 314, 568], 
        [403, 462, 316, 567], [485, 547, 318, 572], [573, 637, 320, 571],
        [656, 718, 319, 571], [742, 802, 320, 574], [825, 889, 326, 578],
        [906, 975, 324, 576], [994, 1056, 327, 578]]

for i in range(10) :
    coords = []
    print("number", i)

    for k in range(4) :
        coords.append(int(coords_origin[i][k]*0.5))

    hurt_modi = hurt.copy()
    hurt_modi = hurt[coords[2]:coords[3], coords[0]:coords[1]]

    result = cv2.matchTemplate(dst, hurt_modi, cv2.TM_SQDIFF_NORMED)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc
    h, w = hurt_modi.shape[0:2]

    coords_result = [x, x+w, y, y+h]
    print("original coordinates : ", end='')
    print(coords)

    for k in range(4) :
        print("predicted coordinates :", coords_result[k], end=' ')
        if not check(coords[k], coords_result[k], 0.03) :
            print("This is faulty!")

    cv2.putText(dst, str(i), (coords[0], coords[2]+10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
    cv2.putText(dst2, str(i), (x, y+10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
    dst = cv2.rectangle(dst, (coords[0], coords[2]), (coords[1], coords[3]) , (0, 0, 255), 1)
    dst2 = cv2.rectangle(dst2, (x, y), (x +  w, y + h) , (0, 0, 255), 1)

addh = cv2.hconcat([dst2, dst])
cv2.imshow("dst", addh)
cv2.waitKey(0)
cv2.destroyAllWindows()