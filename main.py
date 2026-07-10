import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path="hand_landmarker.task"),num_hands=2,running_mode=VisionRunningMode.VIDEO)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
time = 0

#hand detect gesture 
def is_open(hand):
    tips = [8, 12, 16, 20]
    mids = [6, 10, 14, 18]
    up = 0
    for t, m in zip(tips, mids):
        if hand[t].y < hand[m].y:
            up += 1
    return up >= 4

#hand detect gesture metal
def is_metal(hand):
    thumb = abs(hand[4].x -hand[3].x) > 0.03
    index = (hand[8].y <hand[6].y)
    middle = (hand[12].y >hand[10].y)
    ring = (hand[16].y >hand[14].y)
    pinky = (hand[20].y <hand[18].y)
    return (thumb and index and middle and ring and pinky)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    result = detector.detect_for_video(image,time)
    time += 1
    out = frame.copy()
    open_count = 0
    metal_count = 0
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = []
            for lm in hand:
                x = int(lm.x *frame.shape[1])
                y = int(lm.y *frame.shape[0])
                pts.append((x, y))
                cv2.circle(out,(x, y),4,(255,255,255),-1)
            lines = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for a,b in lines:
                cv2.line(out,pts[a],pts[b],(255,255,255),2)
            if is_open(hand):
                open_count += 1
            if is_metal(hand):
                metal_count += 1
    #green
    if metal_count >= 2:
        green = np.full_like(out,(80,255,80))
        out = cv2.addWeighted(green,0.55,out,0.45,0)
        cv2.putText(out,"GREEN GREEN",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    #red
    elif open_count >= 2:
        red = np.full_like(out,(80,80,255))
        out = cv2.addWeighted(red,0.55,out,0.45,0)
        cv2.putText(out,"RED RED",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    cv2.putText(out,"ESC / Q to quit",(20,470),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.imshow("REDRED - Gesture Camera",out)
    key = cv2.waitKey(1)
    if (key == 27 or key == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()