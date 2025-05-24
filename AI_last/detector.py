# detector.py
# YOLO와 OpenPose를 이용해 객체 탐지와 포즈 추정을 수행하는 모듈
import torch
import cv2
from ultralytics import YOLO
from src.body import Body

class Detector:
    def __init__(self, yolo_model_path='yolov9m.pt', openpose_model_path='model/body_pose_model.pth'):
        # YOLOv9 객체 탐지 모델 초기화
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(yolo_model_path).to(device)
        # OpenPose 기반 포즈 추정기 초기화
        self.body_estimation = Body(openpose_model_path)

    def detect_objects(self, frame):
        results = self.model.track(frame, classes=[0], persist=True, verbose=False)
        return results

    def estimate_pose(self, crop):
        return self.body_estimation(crop)
