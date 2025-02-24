#id마다 바운딩박스 색 다르게

from collections import defaultdict
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file
video_path = r"C:\\Users\\kohjj\\ssolitim\\test video.mp4"
output_path = '%s_tracking_output.mp4' % video_path.split('.')[0]

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 원본 영상의 프레임 속도 및 크기 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Track ID별 색상 저장 딕셔너리
id_to_color = {}

# ID에 따른 색상을 랜덤으로 생성하는 함수
def get_color(track_id):
    if track_id not in id_to_color:
        np.random.seed(track_id)  # 같은 ID면 같은 색상 유지
        id_to_color[track_id] = tuple(np.random.randint(0, 255, 3).tolist())  # 랜덤 색상 생성
    return id_to_color[track_id]

# Loop through the video frames
while cap.isOpened():
    start_time = time.time()
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes=[0], persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 (xyxy 좌표)
        confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도 점수
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표
            confidence = conf  # 신뢰도 점수

            # 트랙 히스토리 저장
            track = track_history[track_id]
            track.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))  # 중심점 저장
            if len(track) > 40:  # 최근 30개 프레임까지만 저장
                track.pop(0)

            # ID별 고유한 색상 가져오기
            color = get_color(track_id)

            # 트래킹 선 그리기 (얇게 설정)
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

            # 바운딩 박스 및 ID 표시 (ID마다 다른 색 적용)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} ({confidence:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 결과 프레임을 파일에 저장
        out.write(frame)

        # 결과 표시
        cv2.imshow('YOLOv8 Person Detection', frame)

        # 프레임 간 실제 지연 시간 계산
        elapsed_time = time.time() - start_time
        delay = max(int((frame_time - elapsed_time) * 1000), 1)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
