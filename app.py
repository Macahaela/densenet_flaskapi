from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import jwt
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import requests
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins on all routes

# Load model
model = load_model("model_densenetRcnn.h5")
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

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def save_to_db(user_id, image_url, image_id, predicted_class, confidence):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        record_id = str(uuid.uuid4())
        now = datetime.now()
        
        query = """
        INSERT INTO retina_history 
            (id, image, "imageId", "predictedClass", "confidenceClass", "createdAt", "updatedAt", "userId")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            record_id, 
            image_url, 
            image_id, 
            predicted_class, 
            confidence, 
            now, 
            now,
            user_id
        ))
        
        cursor.close()
        return record_id
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

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
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            file,
            folder="predictions",
            use_filename=True,
            unique_filename=True,
            overwrite=False,
            transformation=[
                {'width': 500, 'height': 500, 'crop': 'limit'}
            ]
        )
        
        # Get image from Cloudinary URL
        image_url = upload_result['secure_url']
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize(IMG_SIZE)

        # Preprocess image
        img_array = np.array(img) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Save to database with user_id
        image_id = upload_result['public_id']
        record_id = save_to_db(
            user_id=current_user_id,
            image_url=image_url,
            image_id=image_id,
            predicted_class=predicted_class,
            confidence=confidence
        )

        return jsonify({
            'success': True,
            'user_id': current_user_id,
            'record_id': record_id,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence
            },
            'image': {
                'public_id': upload_result['public_id'],
                'url': image_url,
                'format': upload_result['format'],
                'width': upload_result['width'],
                'height': upload_result['height']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predicted/guest', methods=['POST'])
def predict_guest():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    try:
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            file,
            folder="guest_predictions",
            use_filename=True,
            unique_filename=True,
            overwrite=False,
            transformation=[
                {'width': 500, 'height': 500, 'crop': 'limit'}
            ]
        )
        
        # Preprocess image
        image_url = upload_result['secure_url']
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize(IMG_SIZE)

        # Predict
        img_array = np.array(img) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Save to guest history table
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            record_id = str(uuid.uuid4())
            now = datetime.now()
            
            query = """
            INSERT INTO retina_history_guest 
                (id, image, "imageId", "predictedClass", "confidenceClass", "createdAt", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                record_id, 
                image_url, 
                upload_result['public_id'], 
                predicted_class, 
                confidence, 
                now, 
                now
            ))
            
            cursor.close()
            
            return jsonify({
                'success': True,
                'record_id': record_id,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence
                },
                'image': {
                    'public_id': upload_result['public_id'],
                    'url': image_url,
                    'format': upload_result['format'],
                    'width': upload_result['width'],
                    'height': upload_result['height']
                }
            })
            
        except Exception as e:
            print(f"Database error: {str(e)}")
            return jsonify({'error': 'Failed to save prediction'}), 500
        finally:
            if conn:
                conn.close()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)