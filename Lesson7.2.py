import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("E:\\C4T\\Lesson6\\haarcascade_frontalface_alt2 (1).xml")
mask = cv2.imread("E:\\C4T\\Lesson6\\6.jpg")
lower = np.array([0,0,0])
higher = np.array([179,255,255])
while True:
    ret, frame = cap.read()

    #convert to gray
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    hsvImage = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    binImg = cv2.inRange(frame,lower,higher)

    #detect face
    faces = cascade.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            # newmask = cv2.resize(mask,(w,h),cv2.INTER_CUBIC)
            # frame[y:y+h,x:x+w,:] = frame[y:y+h,x:x+w,:] - newmask
    cv2.imshow("video", frame)
    key = cv2.waitKey(30)
    if key == ord("q"):
        break
#bai tap: tim` tam ban tay = cach tim` contour ban` tay roi tim` tam^,ve tam bang hinh tron mau` do,lam viec webcam
# viet mot ham` con giu phan` mau` hong` trong anh kitty