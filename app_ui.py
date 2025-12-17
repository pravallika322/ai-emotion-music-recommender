import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from music_mapper import get_music_recommendation
from youtube_player import play_on_youtube

st.set_page_config(
    page_title="AI-Powered Facial Emotion—Based Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        margin-bottom: 3rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #b8b8d4;
        font-weight: 500;
    }
    
    .step-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .step-container:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .step-number {
        display: inline-block;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 60px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .step-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .step-desc {
        font-size: 1rem;
        color: #b8b8d4;
        line-height: 1.6;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.06);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .instruction-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .instruction-box h4 {
        color: #ffffff;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .instruction-box ul {
        color: #e0e0e0;
        line-height: 2;
        margin: 0;
    }
    
    .instruction-box li {
        margin-bottom: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.7rem 1.8rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 5px 20px rgba(56, 239, 125, 0.4);
    }
    
    .status-analyzing {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        animation: pulse-glow 2s ease-in-out infinite;
        box-shadow: 0 5px 20px rgba(245, 87, 108, 0.4);
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 5px 20px rgba(245, 87, 108, 0.4);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 8px 30px rgba(245, 87, 108, 0.6);
        }
    }
    
    .status-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 5px 20px rgba(56, 239, 125, 0.4);
    }
    
    .emotion-result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
        animation: fadeInUp 0.6s ease-out;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .emotion-emoji {
        font-size: 7rem;
        margin: 1rem 0;
        animation: bounceIn 0.8s ease-out;
        filter: drop-shadow(0 10px 20px rgba(0, 0, 0, 0.3));
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .emotion-label {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-container {
        background: rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
    }
    
    .confidence-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .confidence-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    
    .music-recommendation-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.15);
        margin-top: 2rem;
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    
    .music-mood-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .music-message {
        font-size: 1.2rem;
        color: #e0e0e0;
        line-height: 1.8;
        margin-bottom: 2rem;
    }
    
    .genre-section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    
    .genre-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0.5rem;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .genre-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    .waiting-state {
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .waiting-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        opacity: 0.6;
    }
    
    .waiting-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .waiting-desc {
        font-size: 1.2rem;
        color: #b8b8d4;
        line-height: 1.8;
        max-width: 600px;
        margin: 0 auto 2rem auto;
    }
    
    .how-it-works {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .how-it-works h4 {
        color: #ffffff;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    
    .how-it-works ul {
        color: #e0e0e0;
        line-height: 2;
    }
    
    .how-it-works strong {
        color: #667eea;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1.2rem 2rem;
        font-size: 1.3rem;
        font-weight: 700;
        border-radius: 15px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .warning-card {
        background: rgba(255, 152, 0, 0.15);
        border: 2px solid rgba(255, 152, 0, 0.4);
        padding: 2rem;
        border-radius: 15px;
    }
    
    .warning-card h3 {
        color: #ffb74d;
        margin-bottom: 1rem;
    }
    
    .warning-card p {
        color: #e0e0e0;
        line-height: 1.6;
    }
    
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 3rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #888;
    }
    
    .footer p {
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_music_model.h5")

model = load_emotion_model()
labels = ['Angry', 'Happy', 'Neutral', 'Sad']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.markdown("""
    <div class='hero-section'>
        <div class='hero-title'>🎭 AI Emotion-Based Music Recommender</div>
        <div class='hero-subtitle'>Powered by Deep Learning & Computer Vision</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='display: flex; gap: 2rem; margin-bottom: 3rem;'>
        <div style='flex: 1;'>
            <div class='step-container'>
                <div class='step-number'>1</div>
                <div class='step-title'>Capture</div>
                <div class='step-desc'>Take a photo of your facial expression</div>
            </div>
        </div>
        <div style='flex: 1;'>
            <div class='step-container'>
                <div class='step-number'>2</div>
                <div class='step-title'>Analyze</div>
                <div class='step-desc'>AI detects your emotion with CNN</div>
            </div>
        </div>
        <div style='flex: 1;'>
            <div class='step-container'>
                <div class='step-number'>3</div>
                <div class='step-title'>Listen</div>
                <div class='step-desc'>Get personalized music recommendations</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📸 Capture Your Expression</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='instruction-box'>
            <h4>📋 Capture Guidelines</h4>
            <ul>
                <li>Position your face clearly in the camera frame</li>
                <li>Ensure adequate lighting on your face</li>
                <li>Look directly at the camera lens</li>
                <li>Express your current emotion naturally</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    if 'captured' not in st.session_state:
        st.session_state.captured = False
    
    if not st.session_state.captured:
        st.markdown("<div class='status-badge status-ready'>✓ Ready to Capture</div>", unsafe_allow_html=True)
    
    img_file = st.camera_input("Take a picture", label_visibility="collapsed")
    
    if img_file is not None:
        st.session_state.captured = True
        st.markdown("<div class='status-badge status-analyzing'>🔄 AI Analyzing Your Emotion...</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Emotion Detection Results</div>", unsafe_allow_html=True)
    
    if img_file is not None:
        bytes_data = img_file.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.markdown("""
                <div class='warning-card'>
                    <h3>⚠️ No Face Detected</h3>
                    <p>Please adjust your position and ensure your face is clearly visible in the camera frame. Make sure there is adequate lighting.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            (x, y, w, h) = faces[0]
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            preds = model.predict(roi, verbose=0)
            emotion = labels[np.argmax(preds)]
            confidence = np.max(preds)

            st.markdown("<div class='status-badge status-success'>✓ Emotion Successfully Detected</div>", unsafe_allow_html=True)
            
            emotion_emoji = {
                'Angry': '😠',
                'Happy': '😊',
                'Neutral': '😐',
                'Sad': '😢'
            }
            
            st.markdown(f"""
                <div class='emotion-result-card'>
                    <div class='emotion-emoji'>{emotion_emoji.get(emotion, '🎭')}</div>
                    <div class='emotion-label'>{emotion}</div>
                    <div class='confidence-container'>
                        <div class='confidence-label'>Confidence Level</div>
                        <div class='confidence-value'>{confidence:.0%}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(confidence))

            rec = get_music_recommendation(emotion)

            st.markdown(f"""
                <div class='music-recommendation-card'>
                    <div class='music-mood-title'>🎼 {rec['mood']}</div>
                    <div class='music-message'>{rec["message"]}</div>
                    <div class='genre-section-title'>🎸 Recommended Genres</div>
                    <div style='text-align: center;'>
            """, unsafe_allow_html=True)
            
            for keyword in rec["keywords"]:
                st.markdown(f"<span class='genre-chip'>{keyword}</span>", unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("▶️ Play Music on YouTube", use_container_width=True):
                play_on_youtube(rec["keywords"][0])
                st.balloons()
                st.success("🎉 Launching YouTube music player...")

    else:
        st.markdown("""
            <div class='waiting-state'>
                <div class='waiting-icon'>🎵</div>
                <div class='waiting-title'>Waiting for Your Expression</div>
                <div class='waiting-desc'>
                    Your emotion detection results will appear here once you capture an image using the camera on the left.
                </div>
                <div class='how-it-works'>
                    <h4>🤖 How It Works</h4>
                    <ul>
                        <li><strong>Step 1:</strong> Capture your facial expression using the camera</li>
                        <li><strong>Step 2:</strong> AI analyzes your emotion using Convolutional Neural Networks</li>
                        <li><strong>Step 3:</strong> Receive personalized music recommendations based on your mood</li>
                        <li><strong>Step 4:</strong> Play recommended music instantly on YouTube</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div class='footer'>
        <p style='font-size: 1.1rem; color: #b8b8d4;'>Built with ❤️ using TensorFlow, OpenCV & Streamlit</p>
        <p style='font-size: 0.95rem; color: #888;'>Deep Learning | Computer Vision | Music Intelligence</p>
    </div>
""", unsafe_allow_html=True)