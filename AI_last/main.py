# main.py
# 전체 흐름을 통합 실행하는 메인 파일
import cv2
import os
import time
import numpy as np
import threading
import stream_server
from detector import Detector
from classifier import classify_person
from tracker import Tracker
from recorder_manager import RecorderManager
from visualizer import draw_lines
from src import util


# 키포인트 csv파일에 저장
import csv
def save_keypoints_to_single_csv(track_id, frame_index, candidate):
    os.makedirs("keypoints_csv", exist_ok=True)
    filename = f"keypoints_csv/all_keypoints_track_{track_id}.csv"
    
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        # 파일이 처음 생성된 경우 헤더 추가
        if not file_exists:
            writer.writerow(["frame", "index", "x", "y", "confidence"])

#         if isinstance(candidate, np.ndarray):
#             for i in range(candidate.shape[0]):
#                 point = candidate[i][:3]  # x, y, confidence
#                 writer.writerow([frame_index, i, *point])
#         else:
#             writer.writerow([frame_index, "no keypoints", "", "", ""])

# 궤적 및 방향 계산 관련 전역 변수
track_history = {}
MAX_TRAJ_LENGTH = 20

# YOLO, OpenPose 모델 로드
model = Detector()
tracker = Tracker()
recorder_manager = RecorderManager()

# 비디오 경로
#video_path = r"어린이 보호구역 내 도로보행 위험행동 영상/새 폴더/새 폴더/KakaoTalk_20250522_175639451.mp4"
video_path = r"http://192.168.219.121:4747/video"
#video_path = r"https://79d9-106-101-9-161.ngrok-free.app/video"

cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://localhost:5000/video")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("스트리밍 서버 시작")
stream_server.start_server_thread(host='0.0.0.0', port=5000)

# 임시 녹화 삭제 예정
# main.py 안에 cap 설정 이후 추가
import datetime
# 녹화 파일 이름 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
full_recording_path = f"captures/full_recording_{timestamp}.mp4"
os.makedirs("captures", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
full_video_writer = cv2.VideoWriter(full_recording_path, fourcc, fps, (width, height))


# 경계선 설정
# lines = {
#     "GREEN": (width // 2 - 200, width // 2 - 350),
#     "ORANGE": (width // 2 - 50, width // 2 - 50),
#     "RED": (width // 2 + 100, width // 2 + 250),
#     "BLACK": (width // 2 + 220, width // 2 + 550)
# }

# lines = {
#     "GREEN": (width // 2 - 150, width // 2 - 50),
#     "ORANGE": (width // 2, width // 2 + 100),q
#     "RED": (width // 2 + 150, width // 2 + 250),
#     "BLACK": (width // 2 + 300, width // 2 + 400)
# }

# 새로운각도_사람1명.mp4
# lines = {
#     "GREEN": (width // 2 - 500, width // 2 + 160, 660),
#     "ORANGE": (width // 2 - 420, width // 2 + 160, 610),
#     "RED": (width // 2 - 360, width // 2 + 160, 560),
#     "BLACK": (width // 2 - 300, width // 2 + 160, 510)
# }

#/새 폴더/새 폴더/KakaoTalk_20250522_175639451.mp4
# lines = {
#     "GREEN": (width // 2 - 300, width // 2 + 500, 600),
#     "ORANGE": (width // 2 - 250, width // 2 + 450, 560),
#     "RED": (width // 2 - 200, width // 2 + 350, 520),
#     "BLACK": (width // 2 - 150, width // 2 + 200, 480)
# }

#droid cam 사용 시
lines = {
    "GREEN": (width // 2 - 300, width // 2 + 500, int(height * 0.8)),
    "ORANGE": (width // 2 - 250, width // 2 + 450, int(height * 0.75)),
    "RED": (width // 2 - 200, width // 2 + 350, int(height * 0.7)),
    "BLACK": (width // 2 - 150, width // 2 + 200, int(height * 0.65))
}

def record_worker(frame, dominant, x1, y1, x2, y2, lines, frame_index, track_id):
    recorder = recorder_manager.get(track_id)
    recorder.check_escape(frame.copy(), dominant, x1, y1, x2, y2, lines, frame_index)

threads = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    clean_frame = frame.copy()  # 임시 녹화 삭제 예정

    results = model.detect_objects(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    classes = results[0].boxes.cls.cpu().numpy()  # YOLO가 판단한 클래스 (0=Adult, 1=Child)

    for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
        # if conf < 0.5:
        #     continue  # 신뢰도 낮은 객체 무시

        x1, y1, x2, y2 = map(int, box)
        color = tracker.get_color(track_id)

        # if conf >= 0.6:
        #     classification = "Adult" if int(cls) == 0 else "Child"
        # else:
        person_crop = frame[y1:y2, x1:x2].copy()
        if person_crop.size == 0:
            continue

        candidate, subset = model.estimate_pose(person_crop)
        
        # 키포인트 csv파일에 저장
        # if len(candidate) > 0:
        #     save_keypoints_to_single_csv(track_id, frame_index, candidate)

        scale_x = (x2 - x1) / person_crop.shape[1]
        scale_y = (y2 - y1) / person_crop.shape[0]
        for i in range(len(candidate)):
            candidate[i][0] = int(candidate[i][0] * scale_x + x1)
            candidate[i][1] = int(candidate[i][1] * scale_y + y1)

        if len(candidate) == 0:
            classification = "Unknown"
        else:
            bbox_height = y2 - y1
            classification = classify_person(candidate, bbox_height)

        tracker.update_classification(track_id, classification)

        center_point = (int((x1+x2)//2), int((y1+y2)//2))
        track = tracker.update_track(track_id, center_point)
        # track_array = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
        # cv2.polylines(frame, [track_array], isClosed=False, color=color, thickness=2)

        # counts = tracker.id_to_class_counter[track_id]
        # dominant = tracker.get_dominant_classification(track_id)
        # label = f"ID:{track_id} {dominant} (C:{counts['Child']}, A:{counts['Adult']}, U:{counts['Unknown']})"
        ###################################################################################################

        # 궤적 관리
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(center_point)
        if len(track_history[track_id]) > MAX_TRAJ_LENGTH:
            track_history[track_id].pop(0)

        smoothed = tracker.smooth_trajectory(track_history[track_id])
        direction = tracker.get_direction(smoothed)

        track_array = np.array(smoothed, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [track_array], isClosed=False, color=color, thickness=2)

        counts = tracker.id_to_class_counter[track_id]
        dominant = tracker.get_dominant_classification(track_id)
        label = f"ID:{track_id} {dominant} (C:{counts['Child']}, A:{counts['Adult']}, U:{counts['Unknown']}) Dir:{direction}"

        ##################################################################################################################

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        #if conf < 0.5 and 'candidate' in locals() and len(candidate) > 0:
        frame = util.draw_bodypose(frame, candidate, subset)
        
        # # 어른과 어린이가 프레임에 같이 있으면 이탈 감지 중지
        # has_adult = any(
        #     tracker.get_dominant_classification(tid) == 'Adult'
        #     for tid in track_ids
        # )
        # if dominant == 'Child' and not has_adult:
        #     t = threading.Thread(target=record_worker, args=(frame, dominant, x1, y1, x2, y2, lines, frame_index, track_id))
        #     t.start()
        #     threads.append(t)
            
        # 어른과 어린이가 프레임에 같이 있어도 이탈 감지
        if dominant == 'Child':
            t = threading.Thread(target=record_worker, args=(frame, dominant, x1, y1, x2, y2, lines, frame_index, track_id))
            t.start()
            threads.append(t)
        
    active_track_ids = set(track_ids)  # 현재 프레임에 감지된 ID들

    # 추가: 추적 안 된 ID들도 녹화 타이머 확인
    for recorder in recorder_manager.recorders.values():
        recorder.check_timeout(frame_index, track_id, direction)

    draw_lines(frame, lines, height)
    
    stream_server.shared_frame = frame.copy()
    cv2.imshow('DeepSORT + OpenPose', frame)
    
    frame_index += 1

    full_video_writer.write(clean_frame)  # 임시 녹화 삭제 예정
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# 임시 녹화 삭제 예정
full_video_writer.release()
print(f"전체 영상 저장 완료: {full_recording_path}")

# 모든 녹화 스레드가 끝날 때까지 대기
for t in threads:
    t.join()

cv2.destroyAllWindows()