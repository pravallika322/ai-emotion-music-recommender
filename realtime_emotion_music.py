import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

from music_mapper import get_music_recommendation
from youtube_player import play_on_youtube

# ==========================
# 1. LOAD MODEL & LABELS
# ==========================
model = load_model("emotion_music_model.h5")
labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# ==========================
# 2. FACE DETECTOR
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# 3. STABILITY SETTINGS
# ==========================
STABLE_TIME_REQUIRED = 3.0  # seconds
last_emotion = None
emotion_start_time = None
music_played = False

# ==========================
# 4. START CAMERA
# ==========================
cap = cv2.VideoCapture(0)
print("🎥 Camera started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    if len(faces) == 0:
        # No face detected → reset
        last_emotion = None
        emotion_start_time = None
        music_played = False

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        preds = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(preds)]
        confidence = np.max(preds)

        # ==========================
        # 5. EMOTION STABILITY LOGIC
        # ==========================
        if emotion == last_emotion:
            if emotion_start_time is None:
                emotion_start_time = current_time
            elapsed = current_time - emotion_start_time
        else:
            last_emotion = emotion
            emotion_start_time = current_time
            elapsed = 0
            music_played = False

        # ==========================
        # 6. AUTO MUSIC PLAY
        # ==========================
        if elapsed >= STABLE_TIME_REQUIRED and not music_played:
            rec = get_music_recommendation(emotion)
            keyword = rec["keywords"][0]

            print(f"\n🎭 Stable Emotion: {emotion}")
            print(f"▶ Auto-playing music for: {keyword}")

            play_on_youtube(keyword)
            music_played = True

        # ==========================
        # 7. DISPLAY INFO
        # ==========================
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stable for: {elapsed:.1f}s",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow("Emotion-Aware Music Recommender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
