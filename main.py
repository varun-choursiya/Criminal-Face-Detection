from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import sqlite3
import warnings
import base64
import binascii
from datetime import datetime
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import io
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DATABASE = BASE_DIR / 'criminal_database.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', '0.6'))
FRONTEND_ORIGIN = os.getenv('FRONTEND_ORIGIN', '*')

CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load OpenCV face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("OpenCV Haar cascade failed to load.")
contrib_available = hasattr(cv2, 'face')
if not contrib_available:
    warnings.warn(
        "OpenCV contrib modules are missing (cv2.face). Some face-recognition features may fail."
        " Install 'opencv-contrib-python' in the same Python environment to enable full functionality.",
        RuntimeWarning,
    )

# Database initialization
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS criminals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  crime TEXT NOT NULL,
                  age INTEGER,
                  location TEXT,
                  face_data BLOB NOT NULL,
                  image_path TEXT,
                  date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  criminal_id INTEGER,
                  confidence REAL,
                  detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  location TEXT,
                  FOREIGN KEY(criminal_id) REFERENCES criminals(id))''')
    conn.commit()
    conn.close()

init_db()

class ImageDecodeError(ValueError):
    pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_base64_image(base64_string):
    """Decode base64 image string to numpy array"""
    if not isinstance(base64_string, str) or not base64_string.strip():
        raise ImageDecodeError("Image data is required")

    if ',' in base64_string:
        header, base64_string = base64_string.split(',', 1)
        if header and 'image/' not in header:
            raise ImageDecodeError("Only image uploads are supported")

    try:
        image_data = base64.b64decode(base64_string, validate=True)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    except (binascii.Error, ValueError, UnidentifiedImageError) as exc:
        raise ImageDecodeError("Invalid image data") from exc

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def pil_image_to_bgr(image):
    return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

def image_bytes_to_bgr(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except (ValueError, UnidentifiedImageError) as exc:
        raise ImageDecodeError("Invalid image data") from exc
    return pil_image_to_bgr(image)

def extract_face_features(image):
    """Extract face region and compute histogram features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None, None
    
    # Get largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size
    face_roi = cv2.resize(face_roi, (200, 200))
    
    # Compute histogram
    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return face_roi, hist

def compare_faces(hist1, hist2):
    """Compare two face histograms using correlation"""
    hist1 = np.frombuffer(hist1, dtype=np.float32)
    hist2 = np.frombuffer(hist2, dtype=np.float32)
    
    # Use correlation method
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def get_json_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Request body must be valid JSON"}), 400)
    return data, None

def parse_optional_age(value):
    if value in (None, ''):
        return None
    try:
        age = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Age must be a whole number") from exc
    if age < 0 or age > 130:
        raise ValueError("Age must be between 0 and 130")
    return age

def criminal_image_url(image_path):
    if not image_path:
        return None
    filename = Path(image_path).name
    return f"/uploads/{filename}"

def serialize_criminal_row(criminal):
    return {
        "id": criminal[0],
        "name": criminal[1],
        "crime": criminal[2],
        "age": criminal[3],
        "location": criminal[4],
        "image_path": criminal[5],
        "image_url": criminal_image_url(criminal[5]),
        "date_added": criminal[6],
    }

def serialize_detection_row(detection):
    return {
        "id": detection[0],
        "criminal_name": detection[1],
        "crime": detection[2],
        "confidence": float(detection[3] or 0),
        "detection_time": detection[4],
        "location": detection[5],
    }

def safe_delete_upload(image_path):
    if not image_path:
        return

    upload_root = UPLOAD_FOLDER.resolve()
    candidate = (BASE_DIR / image_path).resolve()
    if upload_root not in candidate.parents or not candidate.exists():
        return
    candidate.unlink()

def save_image(image, name):
    filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    if not filename:
        filename = f"criminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = UPLOAD_FOLDER / filename
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError("Failed to save uploaded image")
    return image_path

def add_criminal_record(name, crime, age, location, image):
    face_roi, hist = extract_face_features(image)
    if face_roi is None:
        raise ValueError("No face detected in image. Please use a clear frontal face photo.")

    image_path = save_image(image, name)

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        '''INSERT INTO criminals (name, crime, age, location, face_data, image_path)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (name, crime, age, location, hist.tobytes(), str(image_path)),
    )
    criminal_id = c.lastrowid
    conn.commit()
    conn.close()
    return {
        "message": "Criminal added successfully",
        "criminal_id": criminal_id,
        "name": name,
    }

def detect_criminal_record(image, detection_location='Unknown'):
    face_roi, hist = extract_face_features(image)
    if face_roi is None:
        return {"message": "No face detected in image", "match": False}

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT id, name, crime, age, location, face_data FROM criminals')
    criminals = c.fetchall()

    if len(criminals) == 0:
        conn.close()
        return {"message": "No criminals in database", "match": False}

    best_match = None
    best_similarity = 0

    for criminal in criminals:
        criminal_id, name, crime, age, crim_location, stored_hist = criminal
        similarity = compare_faces(stored_hist, hist.tobytes())
        if similarity > MATCH_THRESHOLD and similarity > best_similarity:
            best_similarity = similarity
            best_match = {
                "id": criminal_id,
                "name": name,
                "crime": crime,
                "age": age,
                "location": crim_location,
                "confidence": round(similarity * 100, 2),
            }

    if best_match:
        c.execute(
            '''INSERT INTO detections (criminal_id, confidence, location)
               VALUES (?, ?, ?)''',
            (best_match['id'], best_match['confidence'], detection_location),
        )
        conn.commit()
        conn.close()
        return {"match": True, "criminal": best_match}

    conn.close()
    return {"match": False, "message": "No criminal match found in database"}

def list_criminals():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT id, name, crime, age, location, image_path, date_added FROM criminals')
    criminals = [serialize_criminal_row(row) for row in c.fetchall()]
    conn.close()
    return criminals

def list_detections(limit=50):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        '''SELECT d.id, c.name, c.crime, d.confidence, d.detection_time, d.location
           FROM detections d
           JOIN criminals c ON d.criminal_id = c.id
           ORDER BY d.detection_time DESC
           LIMIT ?''',
        (limit,),
    )
    detections = [serialize_detection_row(row) for row in c.fetchall()]
    conn.close()
    return detections

def get_criminal_record(criminal_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT id, name, crime, age, location, image_path, date_added FROM criminals WHERE id = ?',
        (criminal_id,),
    )
    criminal = c.fetchone()
    if not criminal:
        conn.close()
        return None

    c.execute(
        'SELECT confidence, detection_time, location FROM detections WHERE criminal_id = ? ORDER BY detection_time DESC',
        (criminal_id,),
    )
    detections = c.fetchall()
    conn.close()

    result = serialize_criminal_row(criminal)
    result["detections"] = [{"confidence": d[0], "time": d[1], "location": d[2]} for d in detections]
    return result

def delete_criminal_record(criminal_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT image_path, name FROM criminals WHERE id = ?', (criminal_id,))
    result = c.fetchone()
    if not result:
        conn.close()
        raise LookupError("Criminal not found")

    c.execute('DELETE FROM detections WHERE criminal_id = ?', (criminal_id,))
    c.execute('DELETE FROM criminals WHERE id = ?', (criminal_id,))
    conn.commit()
    conn.close()

    safe_delete_upload(result[0])
    return {"message": "Criminal deleted successfully", "name": result[1]}

@app.route('/')
def home():
    return jsonify({
        "message": "Criminal Face Detection API (OpenCV Edition)",
        "version": "2.0 - Python 3.13 Compatible",
        "endpoints": {
            "/add_criminal": "POST - Add new criminal to database",
            "/detect": "POST - Detect criminal from image",
            "/criminals": "GET - List all criminals",
            "/detections": "GET - Get detection history",
            "/criminal/<id>": "GET - Get criminal details"
        }
    })

@app.route('/add_criminal', methods=['POST'])
def add_criminal():
    try:
        data, error_response = get_json_payload()
        if error_response:
            return error_response
        
        name = (data.get('name') or '').strip()
        crime = (data.get('crime') or '').strip()
        try:
            age = parse_optional_age(data.get('age'))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        location = (data.get('location') or '').strip() or None
        image_base64 = data.get('image')
        
        if not all([name, crime, image_base64]):
            return jsonify({"error": "Name, crime, and image are required"}), 400
        
        try:
            image = decode_base64_image(image_base64)
        except ImageDecodeError as exc:
            return jsonify({"error": str(exc)}), 400
        
        try:
            result = add_criminal_record(name, crime, age, location, image)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_criminal():
    try:
        data, error_response = get_json_payload()
        if error_response:
            return error_response

        image_base64 = data.get('image')
        detection_location = (data.get('location') or 'Unknown').strip()
        
        if not image_base64:
            return jsonify({"error": "Image is required"}), 400
        
        try:
            image = decode_base64_image(image_base64)
        except ImageDecodeError as exc:
            return jsonify({"error": str(exc)}), 400
        
        result = detect_criminal_record(image, detection_location)
        return jsonify(result), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/criminals', methods=['GET'])
def get_criminals():
    try:
        criminals = list_criminals()
        response_items = [{k: v for k, v in criminal.items() if k != "image_path"} for criminal in criminals]
        return jsonify({"criminals": response_items, "count": len(response_items)}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detections', methods=['GET'])
def get_detections():
    try:
        detections = list_detections()
        return jsonify({"detections": detections, "count": len(detections)}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/criminal/<int:criminal_id>', methods=['GET'])
def get_criminal(criminal_id):
    try:
        criminal = get_criminal_record(criminal_id)
        if not criminal:
            return jsonify({"error": "Criminal not found"}), 404

        criminal.pop("image_path", None)
        return jsonify(criminal), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_criminal/<int:criminal_id>', methods=['DELETE'])
def delete_criminal(criminal_id):
    try:
        try:
            result = delete_criminal_record(criminal_id)
        except LookupError:
            return jsonify({"error": "Criminal not found"}), 404
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))


@app.route('/health', methods=['GET'])
def health_check():
    # Basic health + DB connectivity check
    db_exists = DATABASE.exists()
    db_connected = False
    criminal_count = None
    try:
        conn = sqlite3.connect(DATABASE, timeout=3)
        c = conn.cursor()
        c.execute("SELECT count(1) FROM criminals")
        row = c.fetchone()
        criminal_count = int(row[0]) if row else 0
        conn.close()
        db_connected = True
    except Exception:
        db_connected = False

    return jsonify({
        "status": "ok",
        "version": "2.0",
        "database_present": bool(db_exists),
        "database_connected": db_connected,
        "criminal_count": criminal_count,
    }), 200

if __name__ == '__main__':
    print("Criminal Face Detection System Starting...")
    print("Using OpenCV for face detection (Python 3.13 compatible)")
    print("Database initialized")
    print("Face recognition ready")
    print("Server running on http://0.0.0.0:" + os.getenv('PORT', '5000'))
    print("\nQuick Start:")
    print("   1. Add criminals via /add_criminal endpoint")
    print("   2. Detect faces via /detect endpoint")
    print("   3. View all data via /criminals and /detections")
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', '5000')),
        debug=os.getenv('FLASK_DEBUG', '0') == '1',
    )
