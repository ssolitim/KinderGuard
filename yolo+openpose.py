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
source = "어린이 보호구역 내 도로보행 위험행동 영상/Training/cctv/driveway_walk/[원천]clip_driveway_walk_2/2020_12_02_10_16_driveway_walk_sun_B_04.mp4"
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

#멀티프로세싱
frame_queue = mp.Queue(maxsize=10)
result_queue = mp.Queue(maxsize=10)

BATCH_SIZE = 4  #배치 사이즈 설정

def classify_person(candidate, bbox_height):
    """ 신체 비율을 기반으로 어른/어린이를 분류 """
    try:
        head = candidate[0][:2]
        shoulder = (candidate[2][:2] + candidate[5][:2]) / 2
        hip = (candidate[8][:2] + candidate[11][:2]) / 2
        ankle = (candidate[10][:2] + candidate[13][:2]) / 2
        
        height = np.linalg.norm(head - ankle)
        head_size = np.linalg.norm(head - shoulder)
        leg_length = np.linalg.norm(hip - ankle)

        R_head = head_size / bbox_height
        R_leg = leg_length / bbox_height

        if R_head < 0.10 or R_leg < 0.10:
            return "Unknown"
        if R_head > 0.30 or R_leg < 0.35:
            return "Child"
        elif R_head < 0.20 or R_leg > 0.42:
            return "Adult"
        else:
            return "Uncertain"
    except:
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

                    #"Child" 또는 "Adult"가 아닐 경우, 이전 값 유지
                    if classification not in ["Child", "Adult"]:
                        classification = person_tracking.get((x1, y1, x2, y2), {}).get('label', "Unknown")

                    #현재 프레임에서 확인된 객체 저장
                    updated_tracking[(x1, y1, x2, y2)] = {'label': classification, 'lost_frames': 0}

                    #객체 ID 기반으로 라벨 표시
                    color = (0, 255, 255) if classification == "Child" else (255, 0, 0)
                    cv2.putText(frame, classification, (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    frame = util.draw_bodypose(frame, candidate, subset)

                    #바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    yolo_process = mp.Process(target=yolo_detect, args=(frame_queue, result_queue))
    openpose_process = mp.Process(target=openpose_detect, args=(result_queue,))
    yolo_process.start()
    openpose_process.start()
    yolo_process.join()
    openpose_process.join()
