import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from music_mapper import get_music_recommendation
from youtube_player import play_on_youtube


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI-Powered Facial Emotion—Based Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# =========================
# GLOBAL STYLES (DARK + CLEAN)
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0b0f19, #111827);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 2.5rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 3rem;
}

.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #e5e7eb;
}

.hero p {
    color: #9ca3af;
    font-size: 1.2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 1rem;
}

.emotion-box {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    padding: 2.5rem;
    border-radius: 22px;
    text-align: center;
    color: white;
}

.genre-chip {
    display: inline-block;
    padding: 0.7rem 1.4rem;
    border-radius: 20px;
    background: #6366f1;
    color: white;
    margin: 0.3rem;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.9rem;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (DO NOT TOUCH)
# =========================
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_music_model.h5")

model = load_emotion_model()
labels = ['Angry', 'Happy', 'Neutral', 'Sad']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "capture"

if "image" not in st.session_state:
    st.session_state.image = None


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <h1>🎭 AI-Powered Facial Emotion—Based Music Recommendation System</h1>
    <p>Powered by Deep Learning & Computer Vision</p>
</div>
""", unsafe_allow_html=True)


# =========================
# PAGE 1 — CAPTURE
# =========================
if st.session_state.page == "capture":

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📸 Capture Your Expression</div>", unsafe_allow_html=True)

        st.info("Ensure good lighting and look directly at the camera.")
        img = st.camera_input("Camera", label_visibility="collapsed")

        if img is not None:
            st.session_state.image = img
            st.success("Image captured successfully")

            if st.button("🔍 Analyze Emotion"):
                st.session_state.page = "result"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>ℹ️ How It Works</div>", unsafe_allow_html=True)
        st.write("""
        1. Capture your facial expression  
        2. AI detects emotion using CNN  
        3. Personalized music is recommended  
        4. Music plays instantly on YouTube
        """)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PAGE 2 — RESULT (DETECTION HERE ONLY)
# =========================
elif st.session_state.page == "result":

    img_file = st.session_state.image

    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("❌ No face detected. Please try again.")

        if st.button("🔁 Go Back"):
            st.session_state.page = "capture"
            st.rerun()

    else:
        with st.spinner("Analyzing facial expression..."):
            (x, y, w, h) = faces[0]
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            preds = model.predict(roi, verbose=0)
            emotion = labels[np.argmax(preds)]
            confidence = float(np.max(preds))

        emoji = {
            "Angry": "😠",
            "Happy": "😊",
            "Neutral": "😐",
            "Sad": "😢"
        }

        st.markdown(f"""
        <div class="emotion-box">
            <div style="font-size:4rem">{emoji.get(emotion)}</div>
            <h2>{emotion}</h2>
            <p>Confidence: {confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

        rec = get_music_recommendation(emotion)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎶 Music Recommendation")
        st.write(f"**Mood:** {rec['mood']}")
        st.write(rec["message"])

        for k in rec["keywords"]:
            st.markdown(f"<span class='genre-chip'>{k}</span>", unsafe_allow_html=True)

        if st.button("▶️ Play on YouTube"):
            play_on_youtube(rec["keywords"][0])

        if st.button("🔁 Try Another Expression"):
            st.session_state.page = "capture"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
