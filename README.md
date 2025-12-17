# 🎭 AI Emotion-Based Music Recommender

An AI-powered facial emotion detection system that analyzes user expressions in real time and recommends personalized music using **Deep Learning**, **Computer Vision**, and **Streamlit**.

---

## 📌 Project Overview

Music has a strong connection with human emotions. This project detects a user's **facial emotion** using a deep learning model and automatically recommends **mood-based music**.

The system captures a face image through a webcam, predicts the emotion using a trained CNN model, maps the emotion to a suitable music mood, and opens relevant music on **YouTube** for an immersive experience.

---

## 🎯 Key Features

- 📸 Real-time facial emotion detection using webcam  
- 🧠 Deep Learning (CNN) based emotion classification  
- 🎵 Emotion → Mood → Music recommendation mapping  
- ▶️ Automatic YouTube music playback  
- 🌐 Interactive and user-friendly **Streamlit web interface**  
- 💻 Resume and demo ready project  

---

## 🧠 Emotions Supported

- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😐 Neutral  

Each detected emotion is mapped to a suitable music mood such as **Energetic**, **Calm**, **Peaceful**, or **Focus**.

---

## 🏗️ System Architecture

1. Capture face image using webcam  
2. Detect face using Haar Cascade (OpenCV)  
3. Preprocess face image (grayscale, resize, normalize)  
4. Predict emotion using CNN model  
5. Map emotion to music mood  
6. Open recommended music on YouTube  

---

## 🛠️ Technologies Used

- **Python 3.9**
- **TensorFlow / Keras** – Deep Learning model
- **OpenCV** – Face detection and image processing
- **NumPy** – Numerical operations
- **Streamlit** – Web application interface
- **YouTube (web browser)** – Music playback

---

## 📁 Project Structure

emotion_music/
│
├── app.py # Main Streamlit application                                                                                                                          
├── app_ui.py # Enhanced UI version                                                                                                                                 
├── preprocessing.py # Image preprocessing logic
├── train_model.py # CNN model training script                                                                                                                      
├── emotion_music_model.h5 # Trained emotion detection model                                                                                                        
├── music_mapper.py # Emotion → music mapping                                                                                                                       
├── youtube_player.py # YouTube playback logic                                                                                                                      
├── realtime_emotion_music.py # Real-time camera emotion detection                                                                                                
├── requirements.txt # Project dependencies                                                                                                                         
├── Dataset/ # Emotion dataset                                                                                                                                      
└── README.md # Project documentatio                                                                                                                               

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pravallika322/ai-emotion-music-recommender.git
cd ai-emotion-music-recommender
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🚀 Run the Application
python -m streamlit run app.py


or (for enhanced UI)

python -m streamlit run app_ui.py


The app will open in your browser at:

http://localhost:8501

📊 Model Training (Optional)

If you want to retrain the model:

python train_model.py


Note: A trained model (emotion_music_model.h5) is already included.

🎥 Demo Flow

Launch the Streamlit app

Capture your facial expression using webcam

Emotion is detected by the AI model

Mood-based music is recommended

Click the button to play music on YouTube

📌 Use Cases

Mental wellness & mood enhancement

Smart music recommendation systems

Human–Computer Interaction (HCI)

AI & ML project demonstrations

Hackathons and academic projects

🌟 Future Enhancements

Real-time emotion stability detection (3–5 seconds)

Spotify / Apple Music integration

Multiple face emotion detection

Emotion history & analytics dashboard

Mobile-friendly deployment

👩‍💻 Author

Pravallika Nidadavolu
AI / ML Enthusiast | Deep Learning | Computer Vision
