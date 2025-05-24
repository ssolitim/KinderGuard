# recorder_manager.py
# 여러 객체를 위한 Recorder 인스턴스를 관리하고 할당하는 모듈
from recorder import Recorder

class RecorderManager:
    def __init__(self):
        self.recorders = {}

    def get(self, track_id):
        if track_id not in self.recorders:
            self.recorders[track_id] = Recorder(track_id)
        return self.recorders[track_id]