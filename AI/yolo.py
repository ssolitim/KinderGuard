import cv2
import torch
import numpy as np
import multiprocessing as mp
import time
import os
import requests
from queue import Empty
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import LoadImages
from src.body import Body
from src import util


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
            #print("head_to_height_ratio : ", head_to_height_ratio)
            #print("leg_to_height_ratio", leg_to_height_ratio)
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
    
    child_count = 0
    adult_count = 0
    is_recording = False
    video_writer = None
    record_black_time = None
    
    while True:
        try:
            batch_results = result_queue.get(timeout=2)
        except Empty:
            continue
        
        updated_tracking = {}

        for frame, boxes in batch_results:
            height, width, _ = frame.shape
            GREEN_LINE_X = width // 2 + 300
            ORANGE_LINE_X = width // 2 + 200
            RED_LINE_X = width // 2 + 100
            BLACK_LINE_X = width // 2
            GREEN_LINE_Y = height // 2 + 100
            ORANGE_LINE_Y = height // 2 + 175
            RED_LINE_Y = height // 2 + 250
            BLACK_LINE_Y = height // 2 + 325
            current_time = time.time()

            for x1, y1, x2, y2, bbox_height in boxes:
                person_crop = frame[y1:y2, x1:x2].copy()

                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    # GPU 메모리 사용량 기준으로 캐시 정리
                    gpu_mem_used = torch.cuda.memory_allocated() / 1024**2  # MB 단위
                    if gpu_mem_used > 2000:
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
                    if classification == "Child":
                        child_count += 1
                    elif classification == "Adult":
                        adult_count += 1
                    #print("Child : ", child_count, "Adult : ", adult_count)
                    
                    # 감지 및 로그 출력 세로
                    if classification == "Child" and BLACK_LINE_X >= x1:
                        print(f"BLACK 경고 - 어린이가 이탈했습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                        if is_recording and record_black_time is None:
                            record_black_time = current_time
                            os.makedirs("captures", exist_ok=True)  # 폴더 없으면 생성
                            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
                            image_filename = os.path.join("captures", f"capture_{timestamp}.jpg")    # captures/파일명 이미지 캡처
                            cv2.imwrite(image_filename, frame)
                            print(f"이미지 저장됨: {image_filename}")
                    elif classification == "Child" and RED_LINE_X >= x1:
                        print(f"RED 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    elif classification == "Child" and ORANGE_LINE_X >= x1:
                        print(f"ORANGE 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                        if not is_recording:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            os.makedirs("videos", exist_ok=True)    # 폴더 없으면 생성
                            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
                            video_filename = os.path.join("videos", f"record_{timestamp}.mp4")   # videos/파일명 동영상 녹화
                            video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
                            is_recording = True
                            record_black_time = None
                            print("녹화 시작!")
                    elif classification == "Child" and GREEN_LINE_X >= x1:
                        print(f"GREEN 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                        if is_recording:
                            print(f"GREEN 복귀 - 녹화 중지 및 영상 삭제")
                            if video_writer is not None:
                                video_writer.release()
                            is_recording = False
                            video_writer = None
                            # 파일 삭제 시도
                            if video_filename and os.path.exists(video_filename):
                                os.remove(video_filename)
                                print(f"{video_filename} 삭제 완료")
                            video_filename = None
                        
                    # 10초간 녹화 후 종료 (Yolo와 OpenPose 사용으로 프레임이 녹화하는 프레임과 달라 정확한 시간을 측정하기 어려움)
                    if is_recording and record_black_time is not None:
                        if current_time - record_black_time > 10:
                            video_writer.release()
                            print("녹화 종료!")
                            is_recording = False
                            video_writer = None
                            record_black_time = None
                    # # 감지 및 로그 출력 가로
                    # if classification == "Child" and RED_LINE_Y <= y1:
                    #     print(f"RED 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    # elif classification == "Child" and ORANGE_LINE_Y <= y1:
                    #     print(f"ORANGE 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")
                    # elif classification == "Child" and GREEN_LINE_Y <= y1:
                    #     print(f"GREEN 경고 - 어린이가 감지되었습니다 좌표(bbox)=({x1}, {y1}, {x2}, {y2})")

                    #객체 ID 기반으로 라벨 표시
                    color = (0, 255, 255) if classification == "Child" else (255, 0, 0)
                    cv2.putText(frame, classification, (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    frame = util.draw_bodypose(frame, candidate, subset)

                    #바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            # 영상 중간 왼쪽에 빨간 세로 선 추가
            cv2.line(frame, (GREEN_LINE_X, 0), (GREEN_LINE_X, height), (22, 219, 29), 2)  # 연두색 두께 2의 직선
            cv2.line(frame, (ORANGE_LINE_X, 0), (ORANGE_LINE_X, height), (0, 94, 255), 2)  # 주황색 두께 2의 직선
            cv2.line(frame, (RED_LINE_X, 0), (RED_LINE_X, height), (0, 0, 255), 2)  # 빨간색 두께 2의 직선
            cv2.line(frame, (BLACK_LINE_X, 0), (BLACK_LINE_X, height), (0, 0, 0), 2)  # 검은색 두께 2의 직선
            
            # #영상 중간 아래에 빨간 가로 선 추가
            # cv2.line(frame, (0, GREEN_LINE_Y), (width, GREEN_LINE_Y), (22, 219, 29), 2)  # 빨간색 두께 2의 직선
            # cv2.line(frame, (0, ORANGE_LINE_Y), (width, ORANGE_LINE_Y), (0, 94, 255), 2)  # 빨간색 두께 2의 직선
            # cv2.line(frame, (0, RED_LINE_Y), (width, RED_LINE_Y), (0, 0, 255), 2)  # 빨간색 두께 2의 직선

        # 녹화 중이면 프레임 저장
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        #업데이트된 객체 정보 유지
        person_tracking = updated_tracking
                
        cv2.namedWindow("YOLOv9 + OpenPose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv9 + OpenPose", 1280, 720)  #원하는 크기로 창 크기 지정
        cv2.imshow("YOLOv9 + OpenPose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey(n) 프레임 조절 100 = 10프레임
            break

    cv2.destroyAllWindows()

# BackEnd 관련 코드
url = "http://13.209.121.22:8080/record/upload"
# 업로드할 파일
# ('필드 이름', ('파일 이름', 파일 객체, 'MIME 타입'))
files = [
    ('uploadFiles', ('image.jpg', open('path/to/image.jpg', 'rb'), 'image/jpeg')),
    ('uploadFiles', ('video.mp4', open('path/to/video.mp4', 'rb'), 'video/mp4'))
]
# 요청 보내기
response = requests.post(url, files=files)
# 응답 확인
print(response.status_code)
print(response.text)

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    yolo_process = mp.Process(target=yolo_detect, args=(frame_queue, result_queue))
    openpose_process = mp.Process(target=openpose_detect, args=(result_queue,))
    yolo_process.start()
    openpose_process.start()
    yolo_process.join()
    openpose_process.join()