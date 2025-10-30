# 🎤 Voice Emotion Recognition System

Real-time voice-based emotion detection using Deep Learning (CNN) trained on IESC dataset. Classifies speech into 5 emotions: **Anger, Fear, Happy, Neutral, Sad**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![React](https://img.shields.io/badge/React-19.2-61dafb.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Dual Input Methods**: Record live audio or upload audio files
- **Real-time Prediction**: Instant emotion classification with confidence scores
- **Browser-based Processing**: WAV conversion using Web Audio API (no FFmpeg required)
- **Mental Health Insights**: Emotion-to-mental-health mapping with suggestions
- **Cross-platform**: Works on Windows, macOS, Linux
- **REST API**: Flask-based ML inference server

---

## 🏗️ System Architecture

```
┌─────────────┐
│   Browser   │
│  (React UI) │
│   Port 3000 │
└──────┬──────┘
       │ POST /predict (Base64 WAV)
       ▼
┌─────────────┐        ┌──────────────┐
│  Flask ML   │◄───────┤  Node.js API │
│   Server    │        │   (Optional) │
│  Port 5000  │        │   Port 8080  │
└──────┬──────┘        └──────────────┘
       │
       ▼
┌─────────────┐
│  CNN Model  │
│ (IESC-trained)│
│  + MFCC     │
└─────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19.2, Web Audio API |
| **Backend** | Flask 3.1, Node.js (Express) |
| **ML Framework** | TensorFlow 2.x, Keras |
| **Audio Processing** | Librosa |
| **Feature Extraction** | MFCC (40 coefficients) |
| **Model** | 1D CNN with 2 Conv blocks |

---

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Git**: For cloning the repository

---

## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ML_MiniProject.git
cd ML_MiniProject
```

### 2️⃣ Setup Python Environment (Flask ML Server)

```bash
cd ML
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**⚠️ Important**: Model files are not included in Git (too large). You have two options:

#### Option 1: Download Pre-trained Model (Recommended)
📥 **Download trained model files from**:
- [Google Drive Link] *(Paste your link below)*
- **Link**: `PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE`

Extract and place these files in `ML/models/`:
- `iesc_cnn_model.h5` (~2MB)
- `scaler.pkl`
- `label_encoder.pkl`

#### Option 2: Train Model Yourself

**Step 1**: Download IESC Dataset
```bash
# Download from Kaggle
https://www.kaggle.com/datasets/ybsingh/indian-emotional-speech-corpora-iesc

# Or use Kaggle API
kaggle datasets download -d ybsingh/indian-emotional-speech-corpora-iesc
unzip indian-emotional-speech-corpora-iesc.zip -d ML_src_code/data/
```

**Step 2**: Extract Features
```bash
cd ML_src_code/scripts
python extract_features_iesc.py
# This creates features_iesc_cnn.pkl
```

**Step 3**: Train Model
```bash
python train_rf_model.py
# This generates:
# - models/iesc_cnn_model.h5
# - models/scaler.pkl
# - models/label_encoder.pkl
```

**Step 4**: Copy models to ML directory
```bash
cp ML_src_code/scripts/models/* ML/models/
```

### 3️⃣ Setup Backend (Node.js - Optional)

```bash
cd ../backend
npm install
```

### 4️⃣ Setup Frontend (React)

```bash
cd ../frontend
npm install
```

---

## 🎯 Usage

### Start All Services

**Terminal 1: Flask ML Server**
```bash
cd ML
python server.py
# Server runs at http://localhost:5000
```

**Terminal 2: Backend API (Optional)**
```bash
cd backend
npm start
# Server runs at http://localhost:8080
```

**Terminal 3: React Frontend**
```bash
cd frontend
npm start
# Opens at http://localhost:3000
```

### Using the Application

1. **Open browser**: Navigate to `http://localhost:3000`
2. **Record Audio**: Click "🎙️ Start Recording" → speak → "⏹️ Stop Recording"
   - OR -
3. **Upload File**: Click "Choose File" → select audio file
4. **Analyze**: Click "🎧 Analyze" button
5. **View Results**: See detected emotion with confidence score
6. **Reset**: Click "🔄 Reset" for new prediction

---

## 🧠 Model Architecture

### CNN Model Layers

```
Input: (40, 1) - MFCC features
   ↓
Conv1D(64, kernel=3, ReLU)
   ↓
MaxPooling1D(pool_size=2)
   ↓
Conv1D(128, kernel=3, ReLU)
   ↓
MaxPooling1D(pool_size=2)
   ↓
Flatten()
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.5)
   ↓
Dense(5, Softmax)
   ↓
Output: [Anger, Fear, Happy, Neutral, Sad]
```

**Training Details**:
- **Dataset**: IESC (Indian Emotional Speech Corpus)
- **Features**: 40 MFCC coefficients
- **Classes**: 5 emotions
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy

---

## 📁 Project Structure

```
ML_MiniProject/
├── ML/                          # Python ML Server
│   ├── models/                  # ⚠️ Gitignored - Download separately
│   │   ├── iesc_cnn_model.h5
│   │   ├── scaler.pkl
│   │   └── label_encoder.pkl
│   ├── venv/                    # Python virtual environment (gitignored)
│   ├── server.py                # Flask API server
│   └── requirements.txt
│
├── backend/                     # Node.js API (optional)
│   ├── node_modules/            # Gitignored
│   ├── server.js
│   └── package.json
│
├── frontend/                    # React UI
│   ├── node_modules/            # Gitignored
│   ├── build/                   # Gitignored
│   ├── src/
│   │   ├── components/
│   │   │   └── VoiceEmotion.jsx
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── .gitignore
│
└── README.md
```

---

## 📡 API Documentation

### Flask ML Server (Port 5000)

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "supported_emotions": ["Anger", "Fear", "Happy", "Neutral", "Sad"]
}
```

#### `POST /predict`
Predict emotion from audio.

**Request:**
```json
{
  "userId": "user123",
  "audioData": "UklGRiQAAABXQVZFZm10..." // Base64 WAV
}
```

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.89,
  "mentalHealth": {
    "status": "positive_emotional_state",
    "severity": "none",
    "suggestion": "Keep nurturing positive emotions"
  },
  "timestamp": null,
  "requiresConfirmation": true,
  "modelVersion": "VoiceBasedEmotionClassifier_v1.0"
}
```

---

## 🐛 Troubleshooting

### Issue: Model files not found
```
Solution: Download model files from [link] and place in ML/models/
```

### Issue: Flask server won't start
```
Check Python version: python --version (need 3.8+)
Reinstall dependencies: pip install -r requirements.txt
```

### Issue: React app CORS error
```
Ensure Flask server is running on port 5000
Check CORS is enabled in server.py (line 30)
```

### Issue: Audio recording not working
```
Check browser microphone permissions
Use HTTPS or localhost (required for MediaRecorder API)
```

### Issue: FFmpeg error with audio files
```
This shouldn't happen! The app uses browser-based WAV conversion.
If you see this, ensure you're using the latest code.
```

---

## 📊 Performance

- **Inference Time**: < 200ms per prediction
- **Model Size**: ~2MB (H5 format)
- **Supported Audio**: WAV, MP3, OGG, WebM (auto-converted to WAV)
- **Browser Support**: Chrome, Firefox, Edge, Safari (latest versions)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kshitij Datir**

- GitHub: [@YOUR_GITHUB]
- LinkedIn: [Your LinkedIn]

---

## 🙏 Acknowledgments

- IESC Dataset creators
- TensorFlow/Keras team
- Librosa contributors
- React community

---

## 📞 Support

For issues and questions:
- Open an [Issue](https://github.com/YOUR_USERNAME/ML_MiniProject/issues)
- Email: your.email@example.com

---

**⭐ Star this repo if you find it useful!**

