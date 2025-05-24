# visualizer.py
# 프레임에 경계선을 시각적으로 그려주는 모듈
import cv2

def draw_lines(frame, lines, height):
    cv2.line(frame, (lines["GREEN"][0], lines["GREEN"][2]), (lines["GREEN"][1], lines["GREEN"][2]), (22, 219, 29), 2)
    cv2.line(frame, (lines["ORANGE"][0], lines["ORANGE"][2]), (lines["ORANGE"][1], lines["ORANGE"][2]), (0, 94, 255), 2)
    cv2.line(frame, (lines["RED"][0], lines["RED"][2]), (lines["RED"][1], lines["RED"][2]), (0, 0, 255), 2)
    cv2.line(frame, (lines["BLACK"][0], lines["BLACK"][2]), (lines["BLACK"][1], lines["BLACK"][2]), (0, 0, 0), 2)