import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#각도 계산 함수
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

##모션인식
cap = cv2.VideoCapture(1)

# 모션 횟수 셀 때 변수
counter = 0 
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGTH_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGTH_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGTH_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGTH_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGTH_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGTH_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGTH_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGTH_HIP.value].y]
            
            # Calculate angle
            angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
            angle_l_shoulder = calculate_angle(l_hip, l_shoulder, l_elbow)
            angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
            angle_r_shoulder = calculate_angle(r_hip, r_shoulder, r_elbow)
            
            # Visualize angle
            cv2.putText(image, str(angle_l_elbow), 
                           tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic   (헤엄치는 각도랑 머리 위 아래로 왔다갔다하는 거 찍으면 될듯)
            if angle_l_elbow > 160:
                stage = "down"
            if angle_l_elbow < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
            '''if angle < 45:
                stage = "down"
            if angle'''
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()