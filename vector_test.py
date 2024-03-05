import numpy as np

# v1이 목적지 v2가 배방향
while True :
    vector_1 = [0, -1] #목적지 벡터
    vector_2 = [1, 0] #배방향 벡터

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

    
