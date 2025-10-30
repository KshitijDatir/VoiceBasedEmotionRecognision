import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import os

# Create models directory
if not os.path.exists("models"):
    os.makedirs("models")

# Load features
with open("features_iesc_cnn.pkl", "rb") as f:
    X, y, le = pickle.load(f)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for 1D CNN
X_scaled = X_scaled[..., np.newaxis]

# One-hot encode labels
y_cat = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Build 1D CNN
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model, scaler, and label encoder in /models
model.save("models/iesc_cnn_model.h5")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model, scaler, and label encoder saved in /models")
