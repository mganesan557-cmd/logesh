from flask import Flask, request, jsonify
from datetime import datetime, time
import json
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# In-memory storage for demonstration purposes
# In production, use a proper database
user_activities = []
threat_models = {}
alert_log = []

# Sample user data for demonstration
users = {
    "1001": {"name": "John Doe", "department": "R&D", "access_level": "high"},
    "1002": {"name": "Jane Smith", "department": "HR", "access_level": "medium"},
    "1003": {"name": "Bob Johnson", "department": "Finance", "access_level": "high"},
    "1004": {"name": "Alice Brown", "department": "Marketing", "access_level": "low"}
}

# Sensitive files database
sensitive_files = {
    "F1001": {"name": "product_roadmap.pdf", "sensitivity": "high", "department": "R&D"},
    "F1002": {"name": "employee_records.db", "sensitivity": "high", "department": "HR"},
    "F1003": {"name": "financial_reports.xlsx", "sensitivity": "high", "department": "Finance"},
    "F1004": {"name": "patent_application.txt", "sensitivity": "critical", "department": "R&D"}
}

@app.route('/api/activity', methods=['POST'])
def log_activity():
    """
    Endpoint to log user activity
    Expected JSON payload:
    {
        "user_id": "1001",
        "file_id": "F1001",
        "action": "download",  // download, view, edit, delete
        "timestamp": "2023-05-17T03:45:00Z",
        "data_volume": 5242880  // in bytes, optional for downloads
    }
    """
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['user_id', 'file_id', 'action', 'timestamp']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate user exists
    if data['user_id'] not in users:
        return jsonify({"error": "User not found"}), 404
    
    # Validate file exists
    if data['file_id'] not in sensitive_files:
        return jsonify({"error": "File not found"}), 404
    
    # Add to activities log
    user_activities.append(data)
    
    # Check for potential threats
    threat_score, reasons = evaluate_threat(data)
    
    if threat_score > 0.7:  # High threat threshold
        alert = {
            "timestamp": datetime.now().isoformat(),
            "user_id": data['user_id'],
            "activity": data,
            "threat_score": threat_score,
            "reasons": reasons,
            "status": "new"
        }
        alert_log.append(alert)
        
        # Notify security team (in a real system, this would be email, SMS, etc.)
        notify_security_team(alert)
        
        return jsonify({
            "message": "Activity logged", 
            "threat_detected": True,
            "threat_score": threat_score,
            "alert_id": len(alert_log) - 1
        }), 201
    
    return jsonify({"message": "Activity logged", "threat_detected": False}), 201

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all security alerts"""
    return jsonify({"alerts": alert_log, "count": len(alert_log)})

@app.route('/api/alerts/<int:alert_id>', methods=['PUT'])
def update_alert(alert_id):
    """Update alert status"""
    if alert_id >= len(alert_log):
        return jsonify({"error": "Alert not found"}), 404
    
    data = request.get_json()
    if 'status' not in data:
        return jsonify({"error": "Status field required"}), 400
    
    valid_statuses = ['new', 'investigating', 'resolved', 'false_positive']
    if data['status'] not in valid_statuses:
        return jsonify({"error": f"Status must be one of: {valid_statuses}"}), 400
    
    alert_log[alert_id]['status'] = data['status']
    return jsonify({"message": "Alert updated", "alert": alert_log[alert_id]})

@app.route('/api/analytics/user/<user_id>', methods=['GET'])
def user_analytics(user_id):
    """Get analytics for a specific user"""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    user_acts = [a for a in user_activities if a['user_id'] == user_id]
    
    if not user_acts:
        return jsonify({"error": "No activities found for user"}), 404
    
    # Calculate some basic analytics
    actions_count = defaultdict(int)
    data_volume = 0
    odd_hour_activities = 0
    
    for act in user_acts:
        actions_count[act['action']] += 1
        
        if 'data_volume' in act:
            data_volume += act['data_volume']
        
        # Check if activity occurred during odd hours (10 PM to 6 AM)
        timestamp = datetime.fromisoformat(act['timestamp'].replace('Z', ''))
        if time(22, 0) <= timestamp.time() or timestamp.time() <= time(6, 0):
            odd_hour_activities += 1
    
    return jsonify({
        "user": users[user_id],
        "activity_count": len(user_acts),
        "actions_breakdown": dict(actions_count),
        "total_data_volume": data_volume,
        "odd_hour_activities": odd_hour_activities,
        "activities": user_acts
    })

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train anomaly detection model based on historical data"""
    if not user_activities:
        return jsonify({"error": "No data available for training"}), 400
    
    # Prepare features for machine learning
    # In a real system, this would be much more sophisticated
    features = []
    for activity in user_activities:
        # Extract time-based features
        timestamp = datetime.fromisoformat(activity['timestamp'].replace('Z', ''))
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5  # Saturday or Sunday
        is_odd_hour = 1 if (hour >= 22 or hour <= 6) else 0
        
        # Extract action features
        action_type = activity['action']
        action_encoded = {
            'view': 0,
            'edit': 1,
            'download': 2,
            'delete': 3
        }.get(action_type, 0)
        
        # Extract data volume (normalized)
        data_vol = activity.get('data_volume', 0) / (1024 * 1024)  # Convert to MB
        
        # User access level
        user = users.get(activity['user_id'], {})
        access_level = {
            'low': 0,
            'medium': 1,
            'high': 2
        }.get(user.get('access_level', 'low'), 0)
        
        # File sensitivity
        file_info = sensitive_files.get(activity['file_id'], {})
        sensitivity = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'critical': 3
        }.get(file_info.get('sensitivity', 'low'), 0)
        
        features.append([hour, is_weekend, is_odd_hour, action_encoded, data_vol, access_level, sensitivity])
    
    # Train Isolation Forest model for anomaly detection
    X = np.array(features)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Store the model (in production, this would be saved to persistent storage)
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    threat_models[model_id] = {
        'model': model,
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(X)
    }
    
    return jsonify({
        "message": "Model trained successfully",
        "model_id": model_id,
        "samples": len(X)
    })

def evaluate_threat(activity):
    """Evaluate an activity for potential threats"""
    reasons = []
    threat_score = 0.0
    
    # Check 1: Odd hour activity (10 PM to 6 AM)
    timestamp = datetime.fromisoformat(activity['timestamp'].replace('Z', ''))
    if time(22, 0) <= timestamp.time() or timestamp.time() <= time(6, 0):
        threat_score += 0.3
        reasons.append("Activity during non-working hours")
    
    # Check 2: Large data download
    if activity['action'] == 'download' and 'data_volume' in activity:
        data_volume_mb = activity['data_volume'] / (1024 * 1024)  # Convert to MB
        if data_volume_mb > 100:  # More than 100 MB is suspicious
            threat_score += 0.4
            reasons.append(f"Large data download: {data_volume_mb:.2f} MB")
    
    # Check 3: High sensitivity file access
    file_info = sensitive_files.get(activity['file_id'], {})
    user_info = users.get(activity['user_id'], {})
    
    if file_info.get('sensitivity') in ['high', 'critical']:
        # Check if user has appropriate access level
        user_access = user_info.get('access_level', 'low')
        file_dept = file_info.get('department')
        user_dept = user_info.get('department')
        
        if user_access == 'low' and file_info.get('sensitivity') in ['high', 'critical']:
            threat_score += 0.5
            reasons.append("Low-clearance user accessing high-sensitivity file")
        elif user_dept != file_dept:
            threat_score += 0.2
            reasons.append("User accessing file outside their department")
    
    # Check 4: Delete action on sensitive files
    if activity['action'] == 'delete' and file_info.get('sensitivity') in ['high', 'critical']:
        threat_score += 0.6
        reasons.append("Deletion of sensitive file")
    
    # Apply machine learning model if available
    if threat_models:
        # Use the most recent model
        latest_model_id = max(threat_models.keys())
        model_data = threat_models[latest_model_id]
        model = model_data['model']
        
        # Prepare features for the model
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        is_odd_hour = 1 if (hour >= 22 or hour <= 6) else 0
        
        action_encoded = {
            'view': 0,
            'edit': 1,
            'download': 2,
            'delete': 3
        }.get(activity['action'], 0)
        
        data_vol = activity.get('data_volume', 0) / (1024 * 1024)  # Convert to MB
        
        access_level = {
            'low': 0,
            'medium': 1,
            'high': 2
        }.get(user_info.get('access_level', 'low'), 0)
        
        sensitivity = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'critical': 3
        }.get(file_info.get('sensitivity', 'low'), 0)
        
        features = np.array([[hour, is_weekend, is_odd_hour, action_encoded, data_vol, access_level, sensitivity]])
        
        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = model.predict(features)
        if prediction[0] == -1:
            threat_score += 0.5
            reasons.append("ML model detected anomalous behavior")
    
    # Cap threat score at 1.0
    threat_score = min(threat_score, 1.0)
    
    return threat_score, reasons

def notify_security_team(alert):
    """Simulate notifying security team about a threat"""
    print(f"SECURITY ALERT: User {alert['user_id']} detected with threat score {alert['threat_score']:.2f}")
    print(f"Reasons: {', '.join(alert['reasons'])}")
    # In a real implementation, this would send an email, SMS, or notification
    # to the security team

if __name__ == '__main__':
    app.run(debug=True, port=5000)
