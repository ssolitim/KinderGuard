import cv2
import torch
import numpy as np
import multiprocessing as mp
import time
from queue import Empty
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import LoadImages
from src.body import Body
from src import util
from collections import defaultdict

#장치 선택
device = select_device('0' if torch.cuda.is_available() else 'cpu')

#Yolo 모델 불러오기
weights = Path('yolov9m.pt')
model = DetectMultiBackend(weights, device=device, fp16=True)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((320, 320), s=stride)

#Openpose 모델 불러오기
body_estimation = Body('model/body_pose_model.pth')

person_tracking = {}
MAX_LOST_FRAMES = 10

#영상 파일 경로
#source = "어린이 보호구역 내 도로보행 위험행동 영상/Training/cctv/driveway_walk/[원천]clip_driveway_walk_4_8/2020_11_16_12_31_driveway_walk_sun_A_5.mp4"
#source = "어린이 보호구역 내 도로보행 위험행동 영상/Training/cctv/driveway_walk/[원천]clip_driveway_walk_2/2020_12_02_10_16_driveway_walk_sun_B_04.mp4"
source = "어린이 보호구역 내 도로보행 위험행동 영상/새 폴더/20250314_094123.mp4"
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

#멀티프로세싱
frame_queue = mp.Queue(maxsize=10)
result_queue = mp.Queue(maxsize=10)

BATCH_SIZE = 4  #배치 사이즈 설정

def classify_person(candidate, bbox_height):
    """ 신체 비율을 기반으로 어른/어린이를 분류 - Unknown 결과 최소화 """
    try:
        # 필요한 키포인트 추출
        head = candidate[0][:2]  # 코(nose)
        
        # 최소한의 필수 키포인트 확인 (목, 어깨, 골반, 발목)
        # OpenPose 키포인트 인덱스: 1=목, 2/5=어깨, 8/11=골반, 10/13=발목
        essential_points = [1, 2, 5, 8, 11, 10, 13]
        valid_points = sum(1 for i in essential_points if not np.isnan(candidate[i][0]))
        
        # 목 키포인트가 있으면 사용, 없으면 어깨 중간점 위쪽으로 추정
        if not np.isnan(candidate[1][0]).any():
            neck = candidate[1][:2]
        elif not np.isnan(candidate[2][0]).any() and not np.isnan(candidate[5][0]).any():
            # 어깨 중간점 위로 목 위치 추정
            r_shoulder = candidate[2][:2]
            l_shoulder = candidate[5][:2]
            shoulder_mid = (r_shoulder + l_shoulder) / 2
            # 목은 어깨 중간에서 조금 위쪽으로 추정
            neck = shoulder_mid - np.array([0, 0.05 * bbox_height])
        else:
            return "Unknown"
        
        # 어깨 중심점 계산 (한쪽이라도 있으면 사용)
        if not np.isnan(candidate[2][0]).any() and not np.isnan(candidate[5][0]).any():
            r_shoulder = candidate[2][:2]
            l_shoulder = candidate[5][:2]
            shoulder = (r_shoulder + l_shoulder) / 2
        elif not np.isnan(candidate[2][0]).any():
            shoulder = candidate[2][:2]
        elif not np.isnan(candidate[5][0]).any():
            shoulder = candidate[5][:2]
        else:
            return "Unknown"
        
        # 골반 중심점 계산 (한쪽이라도 있으면 사용)
        if not np.isnan(candidate[8][0]).any() and not np.isnan(candidate[11][0]).any():
            r_hip = candidate[8][:2]
            l_hip = candidate[11][:2]
            hip = (r_hip + l_hip) / 2
        elif not np.isnan(candidate[8][0]).any():
            hip = candidate[8][:2]
        elif not np.isnan(candidate[11][0]).any():
            hip = candidate[11][:2]
        else:
            return "Unknown"
                
        # 발목 중심점 계산 (한쪽이라도 있으면 사용)
        if not np.isnan(candidate[10][0]).any() and not np.isnan(candidate[13][0]).any():
            r_ankle = candidate[10][:2]
            l_ankle = candidate[13][:2]
            ankle = (r_ankle + l_ankle) / 2
        elif not np.isnan(candidate[10][0]).any():
            ankle = candidate[10][:2]
        elif not np.isnan(candidate[13][0]).any():
            ankle = candidate[13][:2]
        else:
            return "Unknown"
        
        # 총 신체 높이 측정
        total_height = np.linalg.norm(head - ankle)
        
        # 주요 비율 계산
        head_size = np.linalg.norm(head - neck)
        torso_length = np.linalg.norm(shoulder - hip)
        leg_length = np.linalg.norm(hip - ankle)
        
        # 신체 비율 계산
        head_to_height_ratio = head_size / total_height
        leg_to_height_ratio = leg_length / total_height
        with open("ratios_log.txt", "a") as log_file:
            log_file.write(f"head_to_height_ratio: {head_to_height_ratio:.4f} | leg_to_height_ratio: {leg_to_height_ratio:.4f}\n")

        
        # 최소 유효 키포인트가 3개 이상이면 판별 시도
        if valid_points >= 3:
            print("head_to_height_ratio : ", head_to_height_ratio)
            print("leg_to_height_ratio", leg_to_height_ratio)
            # 분류 기준 (거리 고려하여 조정)
            # if head_to_height_ratio > 0.3:
            #     return "Unknown"
            # elif leg_to_height_ratio > 0.6:
            #     return "Unknown"
            if head_to_height_ratio > 0.21:
                print("child")
                return "Child"
            elif leg_to_height_ratio > 0.43:
                print("adult")
                return "Adult"
            elif head_to_height_ratio > 0.14:
                print("child")
                return "Child"
            elif head_to_height_ratio < 0.13:
                print("adult")
                return "Adult"
            else:
                # 기본값을 알수없음으로 설정
                return "Unknown"
        else:
            # 기본값을 알수없음으로 설정
            return "Unknown"
    except Exception as e:
        return "Unknown"
    
@smart_inference_mode()
def yolo_detect(frame_queue, result_queue):
    batch_imgs = []
    batch_im0s = []
    
    for path, im, im0s, vid_cap, s in dataset:
        batch_imgs.append(torch.from_numpy(im).float() / 255.0)
        batch_im0s.append(im0s)
        
        if len(batch_imgs) == BATCH_SIZE:
            im_tensor = torch.stack(batch_imgs).to(model.device)
            im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
            
            with torch.no_grad():
                preds = model(im_tensor)
                preds = non_max_suppression(preds, 0.25, 0.45, agnostic=False, max_det=1000)
            
            batch_results = []
            for i, det in enumerate(preds):
                boxes = []
                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], batch_im0s[i].shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:
                            x1, y1, x2, y2 = map(int, xyxy)
                            bbox_height = y2 - y1
                            boxes.append((x1, y1, x2, y2, bbox_height))
                            cv2.rectangle(batch_im0s[i], (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                batch_results.append((batch_im0s[i], boxes))
            result_queue.put(batch_results)
            batch_imgs.clear()
            batch_im0s.clear()

def openpose_detect(result_queue):
    global person_tracking
    
    while True:
        try:
            batch_results = result_queue.get(timeout=2)
        except Empty:
            continue
        
        updated_tracking = {}

        for frame, boxes in batch_results:
            height, width, _ = frame.shape
            GREEN_LINE_X = width // 2 + 200
            ORANGE_LINE_X = width // 2 + 100
            RED_LINE_X = width // 2
            GREEN_LINE_Y = height // 2 + 100
            ORANGE_LINE_Y = height // 2 + 175
            RED_LINE_Y = height // 2 + 250
            
            for x1, y1, x2, y2, bbox_height in boxes:
                person_crop = frame[y1:y2, x1:x2].copy()

                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    torch.cuda.empty_cache()
                    candidate, subset = body_estimation(person_crop)

                    #Keypoints 좌표 변환 (바운딩 박스 기준으로 맞춤)
                    scale_x = (x2 - x1) / person_crop.shape[1]
                    scale_y = (y2 - y1) / person_crop.shape[0]

                    for i in range(len(candidate)):
                        candidate[i][0] = int(candidate[i][0] * scale_x + x1)
                        candidate[i][1] = int(candidate[i][1] * scale_y + y1)

                    #신체 비율 기반으로 분류
                    classification = classify_person(candidate, bbox_height)
                    
                    # 감지 및 로그 출력 세로
                    if classification == "Child" and RED_LINE_X >= x1:
                        print(f"RED 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    elif classification == "Child" and ORANGE_LINE_X >= x1:
                        print(f"ORANGE 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    elif classification == "Child" and GREEN_LINE_X >= x1:
                        print(f"GREEN 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                            
                    # # 감지 및 로그 출력 가로
                    # if classification == "Child" and RED_LINE_Y <= y1:
                    #     print(f"RED 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    # elif classification == "Child" and ORANGE_LINE_Y <= y1:
                    #     print(f"ORANGE 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    # elif classification == "Child" and GREEN_LINE_Y <= y1:
                    #     print(f"GREEN 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")

                    # 현재 프레임에서 확인된 객체 저장
                    updated_tracking[(x1, y1, x2, y2)] = {
                        "label": classification,
                        "lost_frames": 0,
                        "x1": x1  # 현재 x 좌표 저장
                    }

                    #객체 ID 기반으로 라벨 표시
                    color = (0, 255, 255) if classification == "Child" else (255, 0, 0)
                    cv2.putText(frame, classification, (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    frame = util.draw_bodypose(frame, candidate, subset)

                    #바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            # 영상 중간 왼쪽에 빨간 세로 선 추가
            cv2.line(frame, (GREEN_LINE_X, 0), (GREEN_LINE_X, height), (22, 219, 29), 2)  # 빨간색 두께 2의 직선
            cv2.line(frame, (ORANGE_LINE_X, 0), (ORANGE_LINE_X, height), (0, 94, 255), 2)  # 빨간색 두께 2의 직선
            cv2.line(frame, (RED_LINE_X, 0), (RED_LINE_X, height), (0, 0, 255), 2)  # 빨간색 두께 2의 직선
            
            # #영상 중간 아래에 빨간 가로 선 추가
            # cv2.line(frame, (0, GREEN_LINE_Y), (width, GREEN_LINE_Y), (22, 219, 29), 2)  # 빨간색 두께 2의 직선
            # cv2.line(frame, (0, ORANGE_LINE_Y), (width, ORANGE_LINE_Y), (0, 94, 255), 2)  # 빨간색 두께 2의 직선
            # cv2.line(frame, (0, RED_LINE_Y), (width, RED_LINE_Y), (0, 0, 255), 2)  # 빨간색 두께 2의 직선

        #기존에 추적 중이던 객체 중 감지되지 않은 객체 처리
        for key, value in person_tracking.items():
            if key not in updated_tracking:
                value['lost_frames'] += 1
                if value['lost_frames'] < MAX_LOST_FRAMES:  #일정 시간 동안 유지
                    updated_tracking[key] = value

        #업데이트된 객체 정보 유지
        person_tracking = updated_tracking
                
        cv2.namedWindow("YOLOv9 + OpenPose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv9 + OpenPose", 1280, 720)  #원하는 크기로 창 크기 지정
        cv2.imshow("YOLOv9 + OpenPose", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): #waitKey(n) 프레임 조절 100 = 10프레임
            break

    cv2.destroyAllWindows()

# BackEnd 관련 코드
import os
from flask import Flask
from datetime import datetime
from backend.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from backend.database import db
from backend.dbmodels import Record

remote_server = "ubuntu@13.209.121.22"
pem_key = "backend/ssolitim.pem"
port = 22222
remote_image_path = "/home/ubuntu/detect/images/"
remote_video_path = "/home/ubuntu/detect/videos/"

image_file = "static/images/2025-03-27.jpg" # 이미지 파일 예시
video_file = "static/videos/2025-03-27.mp4" # 비디오 파일 예시

# 캡쳐된 이미지와 비디오 파일을 서버로 전송하는 명령어 사용.
os.system(f"scp -i {pem_key} -P {port} {image_file} {remote_server}:{remote_image_path}")
os.system(f"scp -i {pem_key} -P {port} {video_file} {remote_server}:{remote_video_path}")

app = Flask(__name__)

# Flask에 DB 설정 적용
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS

# SQLAlchemy 초기화
db.init_app(app)

def insert_record(image_path, video_path):
    with app.app_context():
        new_record = Record(image_path=image_path, video_path=video_path)
        db.session.add(new_record)
        db.session.commit()
        print(f"Record inserted successfully! Image Path: {image_path}, Video Path: {video_path}")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    yolo_process = mp.Process(target=yolo_detect, args=(frame_queue, result_queue))
    openpose_process = mp.Process(target=openpose_detect, args=(result_queue,))
    yolo_process.start()
    openpose_process.start()
    yolo_process.join()
    openpose_process.join()