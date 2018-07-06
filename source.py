import cv2
import imutils
import numpy as np



def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="int")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def inc_box_size(pts):
    pts = order_points(pts.reshape(4,2))
    w=(int)((pts[0][0]-pts[1][0])*0.2)
    h=(int)(w/4+11)
    pts[0] += [w, -h]
    pts[1] += [-w, -h]
    pts[2] += [-w, h]
    pts[3] += [w, h]
    return pts

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def get_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    ret3, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((0, 1), np.uint8)
    #mask = cv2.dilate(mask,kernel,iterations = 1)

    while False:
        cv2.imshow("aa",mask)
        if cv2.waitKey(5)==27:
            break

    return mask

def is_rect(pnts1):
    pnts=pnts1.reshape(4,2)
    pnts=order_points(pnts)

    for c in pnts1:
        if c not in pnts:
            #print("Fake")
            return False

    top=np.arctan2((pnts[0][0]-pnts[1][0]),(pnts[0][1]-pnts[1][1]))*180/np.pi
    down=np.arctan2((pnts[3][0]-pnts[2][0]),(pnts[3][1]-pnts[2][1]))*180/np.pi
    left=np.arctan2((pnts[0][0]-pnts[3][0]),(pnts[0][1]-pnts[3][1]))*180/np.pi
    right=np.arctan2((pnts[1][0]-pnts[2][0]),(pnts[1][1]-pnts[2][1]))*180/np.pi

    #print(top,down,left,right)
    if abs(top - down) < 3 or abs(left - right) < 3:
       #print(True)
        return True
    else:
        #print(False)
        return False

def get_number_rect(image):
    hh,ww=image.shape[:2]
    rects=[]
    gamma = [1, 0.5, 2, 0.3, 3]
    for g in gamma:
        img = adjust_gamma(image, g)
        cnts = cv2.findContours(get_mask(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the contours
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # cv2.drawContours(image, [c], 0, (0, 0, 255), 1)
            if (len(approx) == 4):
                # cv2.drawContours(image, [c], -1, (0, 0, 255), 1)
                rect = cv2.minAreaRect(approx)
                ((x, y), (w, h), ang) = rect
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if h > 10 and w > 10:
                    if abs(w / h - 5) <1 or abs(h / w - 5) < 1:
                        if (is_rect(approx)):
                            approx=inc_box_size(approx)
                            fake=0
                            for (xx,yy) in approx:
                                if xx<0 or yy<0 or yy>hh or xx>ww:
                                    fake=1
                            if fake:
                                break
                            x,y,w,h=cv2.boundingRect(approx)

                            rects.append(((x+w/2,y+h/2),(w,h),0))

    for r1 in rects:
        for r2 in rects:
            if r1 != r2:
                if cv2.rotatedRectangleIntersection(r1,r2)[0]!=0:
                    rects.remove(r2)
    ans=[]
    for ((x,y),(w,h),ang) in rects:
        ans.append(((int)(x-w/2),(int)(y-h/2),w,h))

    return ans

def get_number_rect_by_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascadePath = "haarcascade_russian_plate_number.xml"
    plateCascade = cv2.CascadeClassifier(cascadePath)
    plates = plateCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return plates

#cap = cv2.VideoCapture(0)


#ret, image = cap.read()
image = cv2.imread("-1.jpg")

image = imutils.resize(image, height=500)

plates1=get_number_rect_by_haar(image)
plates2=get_number_rect(image)

if len(plates1) >= len(plates2):
    plates=plates1
    print("Haar")
else:
    plates=plates2
    print("easyForMe")

for (x,y,w,h) in plates:
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)



while True:
    cv2.imshow('image', image)
    ch = cv2.waitKey(5)
    if ch != -1:
        break
cv2.destroyAllWindows()