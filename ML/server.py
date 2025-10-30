#!/usr/bin/env python3
"""
Voice Emotion Detection Server
Uses your VoiceBasedEmotionClassifier CNN model
"""

import os
import sys
import base64
import tempfile
import librosa
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    tf.config.set_visible_devices([], 'GPU')  # CPU only
except ImportError as e:
    print(f"TensorFlow not installed: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Model files
MODEL_PATH = "models/iesc_cnn_model.h5"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

model = None
scaler = None
label_encoder = None
supported_emotions = []

EMOTION_MENTAL_HEALTH_MAPPING = {
    'Anger': {'status': 'elevated_stress', 'severity': 'moderate', 'suggestion': 'Anger management recommended'},
    'Fear': {'status': 'anxiety_symptoms', 'severity': 'moderate', 'suggestion': 'Try grounding exercises'},
    'Happy': {'status': 'positive_emotional_state', 'severity': 'none', 'suggestion': 'Keep nurturing positive emotions'},
    'Neutral': {'status': 'balanced_emotional_state', 'severity': 'none', 'suggestion': 'Maintain emotional balance'},
    'Sad': {'status': 'low_mood_indicators', 'severity': 'moderate', 'suggestion': 'Reach out to supportive people'}
}

def initialize_model():
    global model, scaler, label_encoder, supported_emotions
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        print("Model files missing!")
        return False
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    supported_emotions = list(label_encoder.classes_)
    print(f"Model loaded! Supported emotions: {supported_emotions}")
    return True

def extract_mfcc(audio_data):
    tmp_path = None
    try:
        audio_bytes = base64.b64decode(audio_data)
        
        # Verify WAV format
        if audio_bytes[:4] != b'RIFF':
            raise ValueError("Audio must be in WAV format")
        
        print(f"Received WAV audio, size: {len(audio_bytes)} bytes")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Load audio - librosa handles WAV natively without ffmpeg
        y, sr = librosa.load(tmp_path, duration=3, sr=None)
        
        # Check if audio was loaded
        if len(y) == 0:
            raise ValueError("Audio file is empty or could not be loaded")
        
        print(f"Loaded audio: {len(y)} samples at {sr} Hz")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        os.unlink(tmp_path)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise Exception(f"Error processing audio: {str(e)}")

def predict_emotion(features):
    features_scaled = scaler.transform([features])
    features_scaled = features_scaled[..., np.newaxis]  # shape (1, 40, 1)
    pred = model.predict(features_scaled, verbose=0)
    idx = np.argmax(pred)
    emotion = label_encoder.inverse_transform([idx])[0]
    confidence = float(pred[0][idx])
    return emotion, confidence

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'supported_emotions': supported_emotions
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'audioData' not in data:
        return jsonify({'error': 'Missing audioData'}), 400
    try:
        print(f"Received audioData of length: {len(data['audioData'])}")
        features = extract_mfcc(data['audioData'])
        print(f"Extracted features shape: {features.shape}")
        
        emotion, confidence = predict_emotion(features)
        print(f"Predicted emotion: {emotion} with confidence: {confidence}")
        
        mental_health = EMOTION_MENTAL_HEALTH_MAPPING.get(emotion, {'status':'unknown','severity':'mild','suggestion':'Take care'})
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'mentalHealth': mental_health,
            'timestamp': None,
            'requiresConfirmation': True,
            'modelVersion': 'VoiceBasedEmotionClassifier_v1.0'
        })
    except Exception as e:
        print(f"ERROR in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not initialize_model():
        sys.exit(1)
    print("Starting ML server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)




