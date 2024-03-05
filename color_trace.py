from collections import deque
from cv2 import contourArea
from matplotlib.pyplot import contour
import numpy as np
import imutils
import cv2


cap = cv2.VideoCapture(0)    
 
while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([138, 120, 20])
    upper_red = np.array([180,255,255])
    lower_yellow = np.array([15,150,100])
    upper_yellow = np.array([35,255,255])


    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    cnts1 = cv2.findContours(mask1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 >50:
            cv2.drawContours(frame,[c],-1,(0,255,0),3)
            M =cv2.moments(c)

            cx = int (M["m10"]/M["m00"])
            cy = int (M["m01"]/M["m00"])
            front = (cx, cy)

            cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
            cv2.putText(frame,"red",(cx-20,cy-20),cv2.FONT_HERSHEY_PLAIN,2.5,(255,255,255),3)
            cv2.putText(frame,str(front),(cx+20,cy+20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),3)
            
           
        
    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 >50:
            cv2.drawContours(frame,[c],-1,(0,255,0),3)
            M =cv2.moments(c)

            cx = int (M["m10"]/M["m00"])
            cy = int (M["m01"]/M["m00"])
            back = (cx,cy)

            cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
            cv2.putText(frame,"yellow",(cx-20,cy-20),cv2.FONT_HERSHEY_PLAIN,2.5,(255,255,255),3)
            cv2.putText(frame,str(back),(cx+20,cy+20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),3)
        

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()
