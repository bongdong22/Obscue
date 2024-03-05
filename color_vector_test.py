from cv2 import contourArea, subtract
from matplotlib.pyplot import contour
import numpy as np
import imutils
import cv2


cap = cv2.VideoCapture(0)

human=(0,0)
front=(0,0)
back=(0,0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=3, minSize=(20,20))
    
    if len(faces) :
        for  x, y, w, h in faces :
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2, cv2.LINE_4)
            cx = int(x + w/2)
            cy = int(y + h/2)
            human = (cx,cy)
            cv2.putText(frame,str(human),(cx+20,cy+20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),3)



    lower_red = np.array([130, 120, 20])
    upper_red = np.array([180,255,255])
    lower_yellow = np.array([15,150, 20])
    upper_yellow = np.array([35,255,255])


    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    cnts1 = cv2.findContours(mask1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 >500:
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
        if area2 >500:
            cv2.drawContours(frame,[c],-1,(0,255,0),3)
            M =cv2.moments(c)

            cx = int (M["m10"]/M["m00"])
            cy = int (M["m01"]/M["m00"])
            back = (cx,cy)

            cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
            cv2.putText(frame,"yellow",(cx-20,cy-20),cv2.FONT_HERSHEY_PLAIN,2.5,(255,255,255),3)
            cv2.putText(frame,str(back),(cx+20,cy+20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),3)


    human_vector = np.array(human)
    front_vector = np.array(front)
    back_vector = np.array(back)

    vector_1 = human_vector-(front_vector+back_vector)/2 #목적지 벡터
    vector_2 = front_vector-back_vector #배방향 벡터

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    rad = np.arccos(dot_product)
    angle_servo = np.rad2deg(rad) #두 벡터 사이각
    direction = np.cross(vector_2, vector_1) #벡터 외적 음수면 우측 양수면 좌측으로

    print(angle_servo)
    if direction<0:
        print("우측으로")
        servo=90+angle_servo  #  서브모터 각도

    elif direction > 0:
        print("좌측으로")
        servo = 90 - angle_servo  #서브모터각도

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(10) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
