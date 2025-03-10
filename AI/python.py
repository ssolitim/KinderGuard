import cv2
import numpy as np
import torch
from src.body import Body
from src import util

# OpenPose 모델 로드
body_estimation = Body('model/body_pose_model.pth')

# YOLOv9 검출된 비디오를 로드
cap = cv2.VideoCapture("yolo_detected.mp4")

# 출력 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("openpose_output.mp4", fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenPose로 키포인트 탐지
    candidate, subset = body_estimation(frame)
    
    # 키포인트를 원본 영상에 그리기
    frame = util.draw_bodypose(frame, candidate, subset)

    out.write(frame)  # 결과 저장
    cv2.imshow("OpenPose Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
