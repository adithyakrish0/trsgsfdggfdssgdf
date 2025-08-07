# app.py (updated)
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from functools import wraps
import os
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder='static')
CORS(app)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medguardian.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
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

# Create database tables
with app.app_context():
    db.create_all()

def get_current_time():
    """Get current time in HH:MM format"""
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"

# Authentication decorator
def authenticate(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        senior_id = request.headers.get('Senior-Id') or request.args.get('seniorId')
        if not senior_id:
            return jsonify({"error": "Authentication required"}), 401
        try:
            senior_id = int(senior_id)
        except ValueError:
            return jsonify({"error": "Invalid senior ID"}), 400
        kwargs['senior_id'] = senior_id
        return func(*args, **kwargs)
    return decorated

# Initialize sample data
def init_sample_data():
    with app.app_context():
        if Senior.query.count() == 0:
            # Create sample senior
            senior = Senior(
                name="Robert Johnson",
                age=72,
                photo="https://randomuser.me/api/portraits/men/75.jpg",
                last_active=datetime.now()
            )
            db.session.add(senior)
            db.session.commit()
            
            # Create medications
            med1 = Medication(
                senior_id=senior.id,
                name="Metformin",
                dosage="500mg",
                type="Diabetes",
                instructions="Take with breakfast and dinner",
                stock=15
            )
            med2 = Medication(
                senior_id=senior.id,
                name="Lisinopril",
                dosage="10mg",
                type="Blood Pressure",
                instructions="Take at noon",
                stock=5
            )
            db.session.add_all([med1, med2])
            db.session.commit()
            
            # Create schedules
            schedule1 = Schedule(
                medication_id=med1.id,
                time="08:00",
                taken=True,
                timestamp=datetime.now() - timedelta(hours=2)
            )
            schedule2 = Schedule(
                medication_id=med1.id,
                time="20:00",
                taken=False
            )
            schedule3 = Schedule(
                medication_id=med2.id,
                time="12:00",
                taken=False
            )
            db.session.add_all([schedule1, schedule2, schedule3])
            db.session.commit()

# Initialize sample data on first run
init_sample_data()

# Static file routes
@app.route('/')
def index():
    """Serve landing page"""
    return send_from_directory('.', 'landing.html')

@app.route('/senior')
def senior_interface():
    """Serve senior interface"""
    return send_from_directory('.', 'senior_interface.html')

@app.route('/caregiver')
def caregiver_dashboard():
    """Serve caregiver dashboard"""
    return send_from_directory('.', 'caregiver.html')

# API endpoints
@app.route('/api/current-medication', methods=['GET'])
@authenticate
def current_medication(senior_id):
    """Get current medication for a senior"""
    current_time = get_current_time()
    
    # Get senior information
    senior = Senior.query.get(senior_id)
    if not senior:
        return jsonify({"error": "Senior not found"}), 404
    
    # Get all medications for senior
    medications = Medication.query.filter_by(senior_id=senior_id).all()
    
    current_meds = []
    for med in medications:
        for sched in med.schedules:
            # Only include upcoming or current medications that haven't been taken
            if not sched.taken and (sched.time == current_time or sched.time > current_time):
                current_meds.append({
                    "medicationId": med.id,
                    "name": med.name,
                    "dosage": med.dosage,
                    "type": med.type,
                    "time": sched.time,
                    "taken": sched.taken,
                    "instructions": med.instructions
                })
    
    # Sort medications by time
    current_meds.sort(key=lambda x: x["time"])
    
    return jsonify({
        "senior": {
            "id": senior.id,
            "name": senior.name,
            "age": senior.age,
            "photo": senior.photo,
            "lastActive": senior.last_active.isoformat() if senior.last_active else None
        },
        "currentMedication": current_meds[0] if current_meds else None,
        "upcomingMedications": current_meds[1:] if len(current_meds) > 1 else []
    })

@app.route('/api/seniors', methods=['GET'])
def get_seniors():
    """Get list of all seniors"""
    seniors_list = Senior.query.all()
    return jsonify([{
        "id": senior.id,
        "name": senior.name,
        "age": senior.age,
        "photo": senior.photo,
        "lastActive": senior.last_active.isoformat() if senior.last_active else None
    } for senior in seniors_list])

@app.route('/api/record-medication', methods=['POST'])
@authenticate
def record_medication(senior_id):
    """Record medication as taken"""
    data = request.get_json()
    medication_id = data.get("medicationId")
    current_time = get_current_time()
    
    medication = Medication.query.get(medication_id)
    if not medication or medication.senior_id != senior_id:
        return jsonify({"error": "Medication not found"}), 404
    
    # Find schedule entry for current time
    schedule_entry = Schedule.query.filter_by(
        medication_id=medication_id,
        time=current_time
    ).first()
    
    if not schedule_entry:
        return jsonify({"error": "No medication scheduled at this time"}), 400
    
    if schedule_entry.taken:
        return jsonify({"error": "Medication already taken"}), 400
    
    # Update the record
    schedule_entry.taken = True
    schedule_entry.timestamp = datetime.now()
    medication.stock -= 1
    
    # Update senior's last active time
    senior = Senior.query.get(senior_id)
    if senior:
        senior.last_active = datetime.now()
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "medication": {
            "id": medication.id,
            "name": medication.name,
            "dosage": medication.dosage,
            "time": schedule_entry.time,
            "timestamp": schedule_entry.timestamp.isoformat()
        }
    })

@app.route('/api/emergency', methods=['POST'])
@authenticate
def emergency_alert(senior_id):
    """Handle emergency alerts"""
    senior = Senior.query.get(senior_id)
    if senior:
        print(f"EMERGENCY ALERT: {senior.name} needs immediate assistance!")
        
        # Update senior's last active time
        senior.last_active = datetime.now()
        db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Emergency alert sent to caregivers",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/vision/session/start', methods=['POST'])
@authenticate
def start_vision_session(senior_id):
    """Start a new medication monitoring session"""
    data = request.get_json()
    medication_id = data.get('medicationId')
    
    session = VisionSession(
        senior_id=senior_id,
        medication_id=medication_id,
        start_time=datetime.now()
    )
    db.session.add(session)
    db.session.commit()
    
    return jsonify({
        "sessionId": session.id,
        "status": "active"
    })

@app.route('/api/vision/event', methods=['POST'])
@authenticate
def record_vision_event(senior_id):
    """Record a vision-detected event"""
    data = request.get_json()
    session_id = data.get("sessionId")
    event_type = data.get("eventType")
    confidence = data.get("confidence", 0.0)
    
    # Verify session exists and belongs to this senior
    session = VisionSession.query.filter_by(id=session_id, senior_id=senior_id).first()
    if not session:
        return jsonify({"error": "Invalid session"}), 404
    
    # Record the event
    event = VisionEvent(
        session_id=session_id,
        event_type=event_type,
        timestamp=datetime.now(),
        confidence=confidence
    )
    db.session.add(event)
    
    # Process the event
    if event_type == "bottle_close":
        # Check if we have a complete sequence: open -> hand_to_mouth -> close
        events = VisionEvent.query.filter_by(session_id=session_id).all()
        has_open = any(e.event_type == "bottle_open" for e in events)
        has_hand_to_mouth = any(e.event_type == "hand_to_mouth" for e in events)
        
        if has_open and has_hand_to_mouth:
            # Complete sequence detected - mark medication as taken
            medication = Medication.query.get(session.medication_id)
            if medication:
                # Find the schedule for this medication at current time
                current_time = get_current_time()
                schedule = Schedule.query.filter_by(
                    medication_id=medication.id,
                    time=current_time
                ).first()
                
                if schedule and not schedule.taken:
                    schedule.taken = True
                    schedule.timestamp = datetime.now()
                    medication.stock -= 1
                    
                    # Update senior's last active time
                    senior = Senior.query.get(senior_id)
                    if senior:
                        senior.last_active = datetime.now()
    
    db.session.commit()
    
    return jsonify({"success": True})

@app.route('/api/vision/session/end', methods=['POST'])
@authenticate
def end_vision_session(senior_id):
    """End a medication monitoring session"""
    data = request.get_json()
    session_id = data.get("sessionId")
    status = data.get("status", "completed")
    
    session = VisionSession.query.filter_by(id=session_id, senior_id=senior_id).first()
    if not session:
        return jsonify({"error": "Invalid session"}), 404
    
    session.end_time = datetime.now()
    session.status = status
    db.session.commit()
    
    return jsonify({"success": True})

@app.route('/api/vision/process-frame', methods=['POST'])
@authenticate
def process_vision_frame(senior_id):
    """Process a video frame using computer vision"""
    data = request.get_json()
    session_id = data.get("sessionId")
    frame_data = data.get("frameData")
    
    # Verify session exists and belongs to this senior
    session = VisionSession.query.filter_by(id=session_id, senior_id=senior_id).first()
    if not session:
        return jsonify({"error": "Invalid session"}), 404
    
    # Decode base64 frame data
    import base64
    from io import BytesIO
    
    # Extract the base64 string
    if 'base64,' in frame_data:
        frame_data = frame_data.split('base64,')[1]
    
    # Decode to bytes
    frame_bytes = base64.b64decode(frame_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame with vision system
    vision_processor = VisionProcessor()
    events = vision_processor.process_frame(frame, session_id)
    
    # Record events
    for event in events:
        vision_event = VisionEvent(
            session_id=session_id,
            event_type=event['type'],
            timestamp=event['timestamp'],
            confidence=event['confidence']
        )
        db.session.add(vision_event)
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "events": events
    })

@app.route('/api/vision/status', methods=['GET'])
def get_vision_status():
    """Get current vision monitoring status for a senior"""
    senior_id = request.args.get('seniorId')
    
    # Find active session for this senior
    session = VisionSession.query.filter_by(
        senior_id=senior_id,
        status='active'
    ).first()
    
    if session:
        return jsonify({
            "status": "Monitoring in progress",
            "sessionId": session.id,
            "startTime": session.start_time.isoformat()
        })
    else:
        return jsonify({
            "status": "Not monitoring"
        })

@app.route('/api/vision/events', methods=['GET'])
def get_vision_events():
    """Get vision events for a senior"""
    senior_id = request.args.get('seniorId')
    limit = request.args.get('limit', 20, type=int)
    
    # Get sessions for this senior
    sessions = VisionSession.query.filter_by(senior_id=senior_id).all()
    session_ids = [session.id for session in sessions]
    
    # Get events for these sessions
    events = VisionEvent.query.filter(
        VisionEvent.session_id.in_(session_ids)
    ).order_by(VisionEvent.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        "events": [{
            "id": event.id,
            "type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "confidence": event.confidence,
            "sessionId": event.session_id
        } for event in events]
    })
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Use PORT environment variable if available
    app.run(host='0.0.0.0', port=port)