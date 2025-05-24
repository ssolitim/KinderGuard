# recorder.py
# 각 객체(track_id)별로 이탈 시 녹화 시작/종료 및 서버 전송을 담당하는 모듈
import os
import time
import cv2
import requests
import json

class Recorder:
    def __init__(self, track_id):
        self.track_id = track_id
        self.is_recording = False   # 현재 녹화 중인지 여부
        self.video_writer = None    # OpenCV VideoWriter 객체
        self.record_black_frame = None  # BLACK 경계선 이탈 시작 프레임 번호
        self.record_stop_frame = None   # 복귀 프레임 번호
        self.record_start_frame = None  # 전체 녹화 시작 프레임
        self.record_frame_count = 0 # 녹화된 프레임 수
        self.target_record_frames = 150  # 최대 녹화 프레임 수 5초 현재 30fps로 설정함 (예: 10초 * 30fps 하지만 성능에 따라 다름)
        self.image_filename = f"captures/image_{track_id}.jpg"  # 썸네일 이미지 저장 경로
        self.video_filename = f"captures/video_{track_id}.mp4"  # 동영상 저장 경로
        self.red_alert_sent = False # RED 경계선 이탈 여부 플래그

    # 저장된 이미지, 영상, 이탈 방향을 서버에 업로드하는 함수
    def upload_to_backend(self, track_id, direction):
        url = "http://13.209.121.22:8080/record/upload"
        files = [
            ('uploadFiles', (os.path.basename(self.image_filename), open(self.image_filename, 'rb'), 'image/jpeg')),
            ('uploadFiles', (os.path.basename(self.video_filename), open(self.video_filename, 'rb'), 'video/mp4')),
            ('data', ('data.json', json.dumps({"trackId": track_id, "direction": direction}), 'application/json'))
        ]
        response = requests.post(url, files=files)
        print(f"[{track_id}] {response.status_code} {response.text}")
    
    # RED 경계선 이탈 시 서버에 알림
    def upload_to_backend_speaker(self, track_id):
        url = "http://13.209.121.22:8080/record/upload"
        payload = {
            "trackId": track_id,
            "alert": "RED"
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(f"[{track_id}] {response.status_code} {response.text}")

    # BLACK 이탈 후 일정 시간이 지나면 녹화를 강제로 종료하고 서버로 업로드
    def check_timeout(self, frame_index, track_id, direction):
        if self.is_recording and self.record_black_frame is not None:
            if frame_index - self.record_black_frame >= self.target_record_frames:
                if self.video_writer:
                    self.video_writer.release()
                print(f"[{self.track_id}] 녹화 종료 - 5초간 이탈 지속")
                self.is_recording = False
                self.video_writer = None
                self.record_black_frame = None
                self.record_stop_frame = None
                if os.path.exists(self.image_filename) and os.path.exists(self.video_filename):
                    self.upload_to_backend(track_id, direction)
                    print(f"{self.image_filename} 삭제 완료")
                    print(f"{self.video_filename} 삭제 완료")
                    self.image_filename = None
                    self.video_filename = None

    # 각 경계선(black/red/orange/green) 이탈 또는 복귀 여부를 판단하고, 녹화 시작/중지 및 서버 알림/업로드 등을 제어함
    def check_escape(self, frame, classification, x1, y1, x2, y2, lines, frame_index):
        # 성인이나 미분류는 무시
        if classification != "Child":
            return

        # BLACK 경계선 이탈
        if lines["BLACK"][2] >= y2 or lines["BLACK"][2] >= y2:
            if self.is_recording and self.record_black_frame is None:
                self.record_black_frame = frame_index
                os.makedirs("captures", exist_ok=True)
                cv2.imwrite(self.image_filename, frame)
                print(f"[{self.track_id}] 이미지 저장됨: {self.image_filename}")

        # RED 경계선 복귀 or 이탈
        elif lines["RED"][2] >= y2 or lines["RED"][2] >= y2:
            if self.is_recording and self.record_black_frame is not None:
                self.record_stop_frame = frame_index
                if self.record_frame_count - self.record_black_frame < 90:
                    if self.video_writer:
                        self.video_writer.release()
                    print(f"[{self.track_id}] 복귀: 녹화 종료 ({self.record_stop_frame - self.record_black_frame:.2f}초)")
                    self.is_recording = False
                    self.red_alert_sent = False
                    self.video_writer = None
                    self.record_black_frame = None
                    if os.path.exists(self.image_filename):
                        os.remove(self.image_filename)
                        print(f"{self.image_filename} 삭제 완료")
                    if os.path.exists(self.video_filename):
                        os.remove(self.video_filename)
                        print(f"{self.video_filename} 삭제 완료")
            elif self.is_recording and not self.red_alert_sent:
                self.upload_to_backend_speaker(self.track_id)
                self.red_alert_sent = True
                print(f"[{self.track_id}] RED 경계선 이탈 경고음 작동")

        # ORANGE 경계선 이탈 또는 복귀
        elif lines["ORANGE"][2] >= y2 or lines["ORANGE"][2] >= y2:
            # 녹화 시작 조건
            if not self.is_recording:
                self.record_start_frame = frame_index
                self.record_frame_count = 0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                os.makedirs("captures", exist_ok=True)
                self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
                self.is_recording = True
                self.record_black_frame = None
                print(f"[{self.track_id}] 녹화 시작: {self.video_filename}")

        # GREEN 경계선 이탈 또는 복귀
        elif lines["GREEN"][2] >= y2 or lines["GREEN"][2] >= y2:
            # 녹화 종료
            if self.is_recording:
                if self.video_writer:
                    self.video_writer.release()
                self.is_recording = False
                self.video_writer = None
                if os.path.exists(self.video_filename):
                    os.remove(self.video_filename)
                    print(f"{self.video_filename} 삭제 완료")
                self.video_filename = None

        # 현재 프레임이 녹화 중이면 기록
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.record_frame_count += 1
        
        # BLACK 경계선 이탈 후 일정 시간 초과 시 녹화 종료
        # if self.is_recording and self.record_black_frame is not None:
        #     if frame_index - self.record_black_frame >= self.target_record_frames:
        #         if self.video_writer:
        #             self.video_writer.release()
        #         print(f"[{self.track_id}] 녹화 종료")
        #         self.is_recording = False
        #         self.video_writer = None
        #         self.record_black_frame = None
        #         self.record_stop_frame = None
        #         if os.path.exists(self.image_filename) and os.path.exists(self.video_filename):
        #             self.upload_to_backend()
        #             os.remove(self.image_filename)
        #             print(f"{self.image_filename} 삭제 완료")
        #             os.remove(self.video_filename)
        #             print(f"{self.video_filename} 삭제 완료")
        #             self.image_filename = None
        #             self.video_filename = None