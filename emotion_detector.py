import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels (order MUST match training)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam (Windows safe)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Webcam started. Press Q to quit.")

# 🔹 Emotion smoothing buffer
emotion_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔹 Improved face detection parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # 🔹 Predict emotion
        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)

        # 🔹 Smooth emotion over last 10 frames
        emotion_history.append(emotion_index)
        if len(emotion_history) > 10:
            emotion_history.pop(0)

        emotion = emotion_labels[
            max(set(emotion_history), key=emotion_history.count)
        ]

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
