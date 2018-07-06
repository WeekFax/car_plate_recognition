import cv2
import imutils
import numpy as np
import os
import pytesseract
from PIL import Image

cascadePath = "haarcascade_russian_plate_number.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

image = cv2.imread("13.jpg")
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    img=gray[y:y+h,x:x+w]
    cv2.imshow("ss",img)
    filename = "test.png"
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(img)
    #os.remove(filename)
    print(text)



gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)



while True:
    cv2.imshow("aaa",image)
    if cv2.waitKey(5)!=-1:
        break