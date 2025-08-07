from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from functools import wraps
import os
from flask_cors import CORS  # Add CORS support

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# In-memory database (consider using a real database in production)
medications = [
    {
        "id": 1,
        "seniorId": 1,
        "name": "Metformin",
        "dosage": "500mg",
        "type": "Diabetes",
        "schedule": [
            {"time": "08:00", "taken": True, "timestamp": "2023-05-15T08:05:00"},
            {"time": "20:00", "taken": False, "timestamp": None}
        ],
        "instructions": "Take with breakfast and dinner",
        "stock": 15
    },
    {
        "id": 2,
        "seniorId": 1,
        "name": "Lisinopril",
        "dosage": "10mg",
        "type": "Blood Pressure",
        "schedule": [
            {"time": "12:00", "taken": False, "timestamp": None}
        ],
        "instructions": "Take at noon",
        "stock": 5
    }
]

seniors = [
    {
        "id": 1,
        "name": "Robert Johnson",
        "age": 72,
        "photo": "https://randomuser.me/api/portraits/men/75.jpg",
        "lastActive": datetime.now().isoformat()
    }
]

def get_current_time():
    """Get current time in HH:MM format"""
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"

def authenticate(func):
    """Authentication decorator for protected endpoints"""
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
    senior_meds = [m for m in medications if m["seniorId"] == senior_id]
    
    current_meds = []
    for med in senior_meds:
        for sched in med["schedule"]:
            if sched["time"] == current_time or (sched["time"] > current_time and not sched["taken"]):
                current_meds.append({
                    "medicationId": med["id"],
                    "name": med["name"],
                    "dosage": med["dosage"],
                    "type": med["type"],
                    "time": sched["time"],
                    "taken": sched["taken"],
                    "instructions": med["instructions"]
                })
    
    current_meds.sort(key=lambda x: x["time"])
    senior = next((s for s in seniors if s["id"] == senior_id), None)
    
    return jsonify({
        "senior": senior,
        "currentMedication": current_meds[0] if current_meds else None,
        "upcomingMedications": current_meds[1:] if len(current_meds) > 1 else []
    })

@app.route('/api/seniors', methods=['GET'])
def get_seniors():
    """Get list of all seniors"""
    return jsonify(seniors)

@app.route('/api/record-medication', methods=['POST'])
@authenticate
def record_medication(senior_id):
    """Record medication as taken"""
    data = request.get_json()
    medication_id = data.get("medicationId")
    current_time = get_current_time()
    
    medication = next((m for m in medications if m["id"] == medication_id and m["seniorId"] == senior_id), None)
    if not medication:
        return jsonify({"error": "Medication not found"}), 404
    
    schedule_entry = next((s for s in medication["schedule"] if s["time"] == current_time), None)
    if not schedule_entry:
        return jsonify({"error": "No medication scheduled at this time"}), 400
    
    if schedule_entry["taken"]:
        return jsonify({"error": "Medication already taken"}), 400
    
    # Update the record
    schedule_entry["taken"] = True
    schedule_entry["timestamp"] = datetime.now().isoformat()
    medication["stock"] -= 1
    
    # Update senior's last active time
    senior = next((s for s in seniors if s["id"] == senior_id), None)
    if senior:
        senior["lastActive"] = datetime.now().isoformat()
    
    return jsonify({
        "success": True,
        "medication": {
            "id": medication["id"],
            "name": medication["name"],
            "dosage": medication["dosage"],
            "time": schedule_entry["time"],
            "timestamp": schedule_entry["timestamp"]
        }
    })

@app.route('/api/emergency', methods=['POST'])
@authenticate
def emergency_alert(senior_id):
    """Handle emergency alerts"""
    senior = next((s for s in seniors if s["id"] == senior_id), None)
    if senior:
        print(f"EMERGENCY ALERT: {senior['name']} needs immediate assistance!")
    
    return jsonify({
        "success": True,
        "message": "Emergency alert sent to caregivers",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Use PORT environment variable if available
    app.run(host='0.0.0.0', port=port)
