# database.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Senior(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    photo = db.Column(db.String(200))
    last_active = db.Column(db.DateTime)

class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    senior_id = db.Column(db.Integer, db.ForeignKey('senior.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    type = db.Column(db.String(50))
    instructions = db.Column(db.Text)
    stock = db.Column(db.Integer)
    
    schedules = db.relationship('Schedule', backref='medication', lazy=True)

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    time = db.Column(db.String(5), nullable=False)  # HH:MM format
    taken = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime)

class VisionSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    senior_id = db.Column(db.Integer, db.ForeignKey('senior.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='active')  # active, completed, failed
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'))
    
class VisionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('vision_session.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # bottle_open, hand_to_mouth, bottle_close, etc.
    timestamp = db.Column(db.DateTime, nullable=False)
    confidence = db.Column(db.Float)
    frame_data = db.Column(db.Text)  # Path to frame image or base64 encoded thumbnail