# classifier.py
# OpenPose 키포인트를 이용해 ‘child’, ‘adult’, ‘unknown’으로 분류하는 모듈
import numpy as np

def classify_person(candidate, bbox_height):
    try:
        # 머리
        head = candidate[0][:2]
        
        essential_points = [1,2,5,8,11,10,13]
        valid_points = sum(1 for i in essential_points if not np.isnan(candidate[i][0]))

        # 목
        if not np.isnan(candidate[1][0]).any():
            neck = candidate[1][:2]
        elif not np.isnan(candidate[2][0]).any() and not np.isnan(candidate[5][0]).any():
            neck = ((candidate[2][:2] + candidate[5][:2]) / 2) - np.array([0, 0.05 * bbox_height])
        else:
            return "Unknown"

        # 엉덩이
        if not np.isnan(candidate[8][0]).any() and not np.isnan(candidate[11][0]).any():
            hip = (candidate[8][:2] + candidate[11][:2]) / 2
        elif not np.isnan(candidate[8][0]).any():
            hip = candidate[8][:2]
        elif not np.isnan(candidate[11][0]).any():
            hip = candidate[11][:2]
        else:
            return "Unknown"

        # 발목
        if not np.isnan(candidate[10][0]).any() and not np.isnan(candidate[13][0]).any():
            ankle = (candidate[10][:2] + candidate[13][:2]) / 2
        elif not np.isnan(candidate[10][0]).any():
            ankle = candidate[10][:2]
        elif not np.isnan(candidate[13][0]).any():
            ankle = candidate[13][:2]
        else:
            return "Unknown"

        # 전체 키, 머리 크기, 다리 길이 계산
        total_height = np.linalg.norm(head - ankle)
        head_size = np.linalg.norm(head - neck)
        leg_length = np.linalg.norm(hip - ankle)

        # 키 대비 머리 크기 비율 및 다리 비율 계산
        head_to_height_ratio = head_size / total_height
        leg_to_height_ratio = leg_length / total_height

        # 유효 포인트가 3개 이상일 때 분류
        if valid_points >= 3:
            if head_to_height_ratio > 0.40:
                return "Child"
            elif leg_to_height_ratio > 2.0:
                return "Child"
            elif leg_to_height_ratio < 1.2:
                return "Adult"
            elif head_to_height_ratio < 0.25:
                return "Adult"
            else:
                return "Unknown"
        else:
            return "Unknown"
    except:
        return "Unknown"