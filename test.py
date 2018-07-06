import cv2 as cv
import cv2
import imutils
import numpy as np

def nothing(x):
    pass



image = cv2.imread("1.jpg")
image= imutils.resize(image, height=600)




cv2.namedWindow('image')

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',5,100,nothing)
cv2.createTrackbar('B','image',6,100,nothing)


while True:
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')

    img=image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret3, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    corners = cv2.goodFeaturesToTrack(gray, r, g/100, b)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    cv2.imshow('dst',img)
    if cv2.waitKey(5) == 27:
        cv2.destroyAllWindows()
        break