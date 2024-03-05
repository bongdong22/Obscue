SB=1 # 종료키 기능을 위한 변수
shot=0 # 목표물 찾았는지 확인하는 변수



from PIL import ImageGrab # 화면 캡쳐
import keyboard # 종료 키

start=(392, 199) # 화면 상에서 카메라 화면 크기 왼쪽 위 (예시)
end=(1501, 956)  # 오른쪽 아래 (예시)
c_blue=(255, 192, 0) # 사람 인식한 네모 박스 rgb
r_blue=(68, 114, 196) # 배 색깔 rgb


while SB==1:
    screen = ImageGrab.grab() # 화면 캡쳐
    for i in range(start[0],end[0],20):
        for j in range(start[1],end[1],20):
            rgb=screen.getpixel((i,j)) # 각 좌표에서의 rgb값 추출
            if abs(rgb[0]-c_blue[0])+abs(rgb[1]-c_blue[1])+abs(rgb[2]-c_blue[2])<80: 
                rgb=screen.getpixel((i+3,j)) # (i+3,j)에서 rgb값 추출
                if abs(rgb[0]-c_blue[0])+abs(rgb[1]-c_blue[1])+abs(rgb[2]-c_blue[2])<80:
                    for k in range(start[0],end[0],20):
                        for m in range(start[1],end[1],20):
                            rgb=screen.getpixel((k,m)) # 각 좌표에서의 rgb값 추출
                            if abs(rgb[0]-r_blue[0])+abs(rgb[1]-r_blue[1])+abs(rgb[2]-r_blue[2])<80: 
                                rgb=screen.getpixel((k+3,m)) # (k+3,m)에서 rgb값 추출
                                if abs(rgb[0]-r_blue[0])+abs(rgb[1]-r_blue[1])+abs(rgb[2]-r_blue[2])<80:
                                    
                                    print(k-i,m-j)
                                    shot=1
                                    break

        if shot==1: # 목표물 찾았다면 for문 빠져나옴
            shot=0
            break
    if keyboard.is_pressed('F3'): # 코드 종료
        SB=0
        break