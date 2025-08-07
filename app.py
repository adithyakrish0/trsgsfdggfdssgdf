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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Use PORT environment variable if available
    app.run(host='0.0.0.0', port=port)