from backend.database import db

class Record(db.Model):
    __tablename__ = 'record'

    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(255), nullable=True)
    video = db.Column(db.String(255), nullable=True)
    date = db.Column(db.TIMESTAMP, nullable=True, server_default=db.func.current_timestamp())
    memo = db.Column(db.String(255), nullable=True)
    is_read = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, image_path, video_path):
        self.image = image_path
        self.video = video_path