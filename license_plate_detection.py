import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertImage(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,100,200)
    return canny

img = cv2.imread("license.jpg")
processed_img = convertImage(img)
original_img = img.copy()

contour_img = processed_img.copy()

contours, heirarchy = cv2.findContours(contour_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for contour in contours:
    p = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.02*p,True)

    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(contour)
        license_img = original_img[y:y+h,x:x+w]
        cv2.imshow("License Detected : ",license_img)
        cv2.drawContours(img, [contour],-1,(0,255,255),3)


cv2.imshow("Image",img)
cv2.waitKey(0)
