from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import jwt
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI ---
# Load model
try:
    model = load_model("model_densenetRcnn.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    # Jika model tidak ada, aplikasi tidak bisa berjalan
    # Sebaiknya hentikan aplikasi jika model gagal dimuat
    exit()

IMG_SIZE = (255, 255)
labels = ['Dr', 'No_Dr']

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET")

# --- KONEKSI DATABASE ---
def get_db_connection():
    """Membuat koneksi ke database."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn

# --- DECORATOR AUTENTIKASI ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'error': 'Bearer token malformed'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user_id = data['userId']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

# --- FUNGSI HELPER ---
def allowed_file(filename):
    """Mengecek apakah ekstensi file diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_and_predict_image(file_stream):
    """Fungsi utama untuk memproses gambar dan melakukan prediksi."""
    # Preprocess image dari file stream di memori
    img = Image.open(file_stream).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return predicted_class, confidence

# --- ENDPOINTS ---
@app.route('/predict', methods=['POST'])
@token_required
def predict(current_user_id):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    try:
        # Baca file ke memori untuk diproses
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        
        # Lakukan prediksi menggunakan helper function
        predicted_class, confidence = process_and_predict_image(in_memory_file)
        
        # Setelah prediksi, upload file ke Cloudinary
        in_memory_file.seek(0)
        upload_result = cloudinary.uploader.upload(
            in_memory_file,
            folder="predictions",
            unique_filename=True
        )
        
        # Simpan ke database
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            record_id = str(uuid.uuid4())
            now = datetime.now()
            
            query = """
            INSERT INTO retina_history (id, image, "imageId", "predictedClass", "confidenceClass", "createdAt", "updatedAt", "userId")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                record_id, upload_result['secure_url'], upload_result['public_id'], 
                predicted_class, confidence, now, now, current_user_id
            ))
            cursor.close()
        finally:
            if conn:
                conn.close()

        return jsonify({
            'success': True,
            'record_id': record_id,
            'prediction': {'class': predicted_class, 'confidence': confidence},
            'image': {'public_id': upload_result['public_id'], 'url': upload_result['secure_url']}
        })
    except Exception as e:
        print(f"Error during prediction for user {current_user_id}: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/predict/guest', methods=['POST'])
def predict_guest():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    try:
        # Baca file ke memori
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        
        # Lakukan prediksi
        predicted_class, confidence = process_and_predict_image(in_memory_file)
        
        # Upload ke Cloudinary
        in_memory_file.seek(0)
        upload_result = cloudinary.uploader.upload(
            in_memory_file,
            folder="guest_predictions",
            unique_filename=True
        )
        
        # Simpan ke database
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            record_id = str(uuid.uuid4())
            now = datetime.now()
            query = """
            INSERT INTO retina_history_guest (id, image, "imageId", "predictedClass", "confidenceClass", "createdAt", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                record_id, upload_result['secure_url'], upload_result['public_id'],
                predicted_class, confidence, now, now
            ))
            cursor.close()
        finally:
            if conn:
                conn.close()
                
        return jsonify({
            'success': True,
            'record_id': record_id,
            'prediction': {'class': predicted_class, 'confidence': confidence},
            'image': {'public_id': upload_result['public_id'], 'url': upload_result['secure_url']}
        })
    except Exception as e:
        print(f"Error during guest prediction: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))