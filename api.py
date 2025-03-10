from flask import Flask, jsonify, send_file, request, abort
import mysql.connector
import os

app = Flask(__name__)

VIDEO_DIR = "static/videos"  # 비디오 파일이 저장된 폴더

def connect_db():
    return mysql.connector.connect(
        host="13.209.121.22",
        user="root",
        password="ssolitim0925",
        database="ssolitim"
    )

"""
API 사용방법
1.
GET/videos 사용
[
    "http://localhost:5000/video?name=video1.mp4",
    "http://localhost:5000/video?name=video2.mp4",
    "http://localhost:5000/video?name=video3.mp4"
]
영상이 3개라면 위와 같이 서버에서 리스트형식으로 영상 url 반환됨.

2. 
GET/http://localhost:5000/video?name=video1.mp4
GET/http://localhost:5000/video?name=video2.mp4
GET/http://localhost:5000/video?name=video3.mp4
위에서 받은 리스트형식의 영상 url를 이용하여 요청날리면 해당하는 영상 파일을 반환
"""
# 저장된 영상 목록 반환 API
@app.route('/videos', methods=['GET'])
def list_videos():
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    video_urls = [f"http://localhost:5000/video?name={f}" for f in files]  # 각 영상 URL 생성
    return jsonify(video_urls)

# 특정 영상 반환 API
@app.route('/video', methods=['GET'])
def get_video():
    filename = request.args.get('name')  # ?name=sample.mp4
    if not filename:
        return abort(400, "No filename provided")

    video_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(video_path):
        return abort(404, "File not found")

    return send_file(video_path, mimetype='video/mp4')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400

    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "회원가입 성공!"}), 201
    except mysql.connector.IntegrityError:
        return jsonify({"error": "이미 존재하는 아이디입니다."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)