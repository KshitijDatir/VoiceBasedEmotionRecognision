import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

data_path = r"..\data\IESC"
features = []
labels = []

for speaker in os.listdir(data_path):
    speaker_path = os.path.join(data_path, speaker)
    if not os.path.isdir(speaker_path):
        continue
    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)
                try:
                    y, sr = librosa.load(file_path, duration=3)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_scaled)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save features, labels, and encoder
with open("features_iesc_cnn.pkl", "wb") as f:
    pickle.dump((X, y_encoded, le), f)

print(f"Feature extraction complete! Total samples: {len(X)}")
