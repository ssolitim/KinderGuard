# tracker.py
# DeepSORT 방식으로 객체를 추적하고, ID별 이동 경로와 분류 기록을 관리하는 모듈
import numpy as np
from collections import defaultdict

class Tracker:
    def __init__(self):
        # track_id 별 이동 경로(history)를 저장 (리스트 형태의 좌표들)
        self.track_history = defaultdict(lambda: [])
        self.id_to_color = {}
        # track_id별로 'Child', 'Adult', 'Unknown' 분류 횟수 저장
        self.id_to_class_counter = defaultdict(lambda: {'Child': 0, 'Adult': 0, 'Unknown': 0})

    def get_color(self, track_id):
        if track_id not in self.id_to_color:
            np.random.seed(track_id)
            self.id_to_color[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.id_to_color[track_id]

    def update_track(self, track_id, center_point):
        track = self.track_history[track_id]
        track.append(center_point)
        if len(track) > 40:
            track.pop(0)
        return track

    # 특정 ID에 대해 'Child', 'Adult', 'Unknown' 중 하나로 분류된 결과를 누적 기록
    def update_classification(self, track_id, classification):
        if classification in ['Child', 'Adult', 'Unknown']:
            self.id_to_class_counter[track_id][classification] += 1

    # 가장 많이 기록된 분류를 반환 (다수결 기준)
    # 동률이거나 데이터 부족 시 'Unknown' 반환
    def get_dominant_classification(self, track_id):
        counts = self.id_to_class_counter[track_id]
        if counts['Child'] > counts['Adult']:
            return 'Child'
        elif counts['Adult'] > counts['Child']:
            return 'Adult'
        else:
            return 'Unknown'
    
    
    # 추적선 보정
    def smooth_trajectory(self, traj, window=7):
        if len(traj) < window:
            return traj
        return [
            (int(np.mean([pt[0] for pt in traj[max(0, i - window + 1):i + 1]])),
            int(np.mean([pt[1] for pt in traj[max(0, i - window + 1):i + 1]])))
            for i in range(len(traj))
        ]
        
    # 방향감지
    def get_direction(self, traj):
        if len(traj) < 6:
            return "stop"
        dx = traj[-1][0] - traj[-5][0]
        dy = traj[-1][1] - traj[-5][1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 5:
            return "stop"
        # 방향 판정 8방향 (시계 방향 시계수 기준)
        angle = np.arctan2(-dy, dx) * 180 / np.pi  # y 반전 보정
        if angle < 0:
            angle += 360

        # 8방향 구간별 이름
        if 337.5 <= angle or angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "right-up"
        elif 67.5 <= angle < 112.5:
            return "straight"
        elif 112.5 <= angle < 157.5:
            return "left-up"
        elif 157.5 <= angle < 202.5:
            return "left"
        elif 202.5 <= angle < 247.5:
            return "left-down"
        elif 247.5 <= angle < 292.5:
            return "back"
        elif 292.5 <= angle < 337.5:
            return "right-down"
        else:
            return "stop"