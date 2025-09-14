import streamlit as st
import numpy as np
import librosa
import joblib
from PIL import Image
from io import BytesIO
import tempfile
import sounddevice as sd
import soundfile as sf
import base64

# === Constants ===
IMG_SIZE = 224
MAX_RECORD_SECONDS = 20

# === Load Pre-trained Models ===
audio_model = joblib.load("audio_emotion_model.pkl")
audio_encoder = joblib.load("audio_label_encoder.pkl")
audio_scaler = joblib.load("audio_scaler.pkl")

# === Emotion Classes ===
audio_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# === Helper Functions ===
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None, res_type="soxr_hq")
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def predict_audio(file_path):
    features = extract_audio_features(file_path)
    features = audio_scaler.transform([features])
    pred = audio_model.predict(features)
    emotion = audio_encoder.inverse_transform(pred)[0]
    return emotion

def record_audio(duration=20, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, fs)
    return temp_file.name

# === Streamlit App Configuration ===
st.set_page_config(page_title="🎧 Audio Emotion Detector", layout="wide")

# Background Image CSS
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6); /* Overlay */
            z-index: -1;
        }}
        .css-18e3th9 {{
            padding-top: 5rem;
            padding-bottom: 5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background(r'C:\praabhass\python\Task 3\Gemini_Generated_Image_qo1cb6qo1cb6qo1c.png')

# App content
st.title("🎧 Audio Emotion Recognition")

option = st.radio("Choose Input Method", ["Upload Audio", "Record Live Audio"])

if option == "Upload Audio":
    audio_file = st.file_uploader("Upload your audio file (WAV)", type=['wav'])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        if st.button("Proceed"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            emotion = predict_audio(tmp_path)
            st.success(f"🎯 Predicted Emotion: {emotion}")

elif option == "Record Live Audio":
    duration = st.slider("Recording Duration (seconds)", min_value=5, max_value=20, value=10)
    if st.button("Record Audio"):
        recorded_path = record_audio(duration=duration)
        st.audio(recorded_path, format='audio/wav')
        emotion = predict_audio(recorded_path)
        st.success(f"🎯 Predicted Emotion: {emotion}")

