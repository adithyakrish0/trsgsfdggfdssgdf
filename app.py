from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from functools import wraps
import os

app = Flask(__name__, static_folder='static')

# Dummy database
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
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"

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

@app.route('/')
def index():
    return send_from_directory('.', 'landing.html')

@app.route('/senior')
def senior():
    return send_from_directory('.', 'senior_interface.html')

@app.route('/caregiver')
def caregiver():
    return send_from_directory('.', 'caregiver.html')

@app.route('/api/current-medication', methods=['GET'])
@authenticate
def current_medication(senior_id):
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
    return jsonify(seniors)

if __name__ == '__main__':
    app.run(port=3000, debug=True)