import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

# 전역 변수로 프레임 저장
shared_frame = None
frame_lock = threading.Lock()

def generate_frames():
    global shared_frame
    
    while True:
        with frame_lock:
            if shared_frame is not None:
                ret, buffer = cv2.imencode('.jpg', shared_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # 약 30 FPS

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def update_frame(frame):
    global shared_frame
    with frame_lock:
        shared_frame = frame.copy() if frame is not None else None

def start_server_thread(host='0.0.0.0', port=5000):
    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, threaded=True),
        daemon=True
    )
    server_thread.start()
    return server_thread