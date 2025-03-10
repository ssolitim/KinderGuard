import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

# Firebase Admin SDK를 초기화
cred = credentials.Certificate('kinderguard-b7db7-firebase-adminsdk-fbsvc-09cc151158.json')
firebase_admin.initialize_app(cred)

# 사용자 FCM 토큰
registration_token = 'deZqu51Q4juhGQqEOiLQIBed0ojom0Y1Mf3oSHnMK6ZO5xPmwaoq' # 예시

# FCM 메시지 전송
message = messaging.Message(
    notification=messaging.Notification(
        title='제목',
        body='FCM 토큰'
    ),
    data={
        'subtitle': '부제목',
        'screen': '4',
    },
    token=registration_token
)

response = messaging.send(message)
print('Successfully sent message:', response)