import streamlit as st
import speech_recognition as sr
import openai
import whisper
from dotenv import load_dotenv
import os
import tempfile
import wave

# Add ffmpeg path so Whisper can find it (adjust if your path is different)
os.environ["PATH"] += os.pathsep + r"C:\Users\Lamis\Desktop\ffmpeg\ffmpeg-7.1.1-essentials_build\bin"

# Load Whisper model once
model = whisper.load_model("base")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

r = sr.Recognizer()

if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

def listen_with_whisper(duration):
    with sr.Microphone() as source:
        try:
            st.info(f"Listening for {duration} seconds... Speak now.")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source, duration=duration)
        except Exception as e:
            st.error(f"Microphone error: {e}")
            return ""

    try:
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
        # Write raw audio data to WAV file properly
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.sample_width)  # Correct sample width from audio object
            wf.setframerate(audio.sample_rate)  # Use actual sample rate from audio object
            wf.writeframes(audio.get_raw_data())

        # Transcribe using Whisper
        result = model.transcribe(wav_path)
        return result["text"]

    except Exception as e:
        st.error(f"Whisper error: {e}")
        return ""

def process_text(text):
    prompt = f"""
You are an AI that processes lecture transcripts.

Step 1: Fix grammar and fill missing words in this raw transcript.
Step 2: Provide a short summary.
Step 3: List bullet points of key ideas.

Transcript:
{text}

Result:
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return "Failed to process the transcript."

# Streamlit UI
st.title("🎧 Live Lecture Summarizer")

st.subheader("⏱️ Choose how long to record")
col1, col2, col3 = st.columns(3)
hours = col1.number_input("Hours", min_value=0, max_value=5, value=0)
minutes = col2.number_input("Minutes", min_value=0, max_value=59, value=0)
seconds = col3.number_input("Seconds", min_value=0, max_value=59, value=5)

total_seconds = hours * 3600 + minutes * 60 + seconds

col4, col5 = st.columns(2)

with col4:
    if st.button("🎙️ Start Listening", disabled=st.session_state.recording):
        st.session_state.recording = True
        with st.spinner(f"Recording for {total_seconds} seconds..."):
            st.session_state.transcript = listen_with_whisper(duration=total_seconds)
        st.session_state.recording = False

with col5:
    if st.button("⏹️ Clear Transcript"):
        st.session_state.transcript = ""

if st.session_state.transcript:
    st.write("### 📝 Transcript")
    st.success(st.session_state.transcript)

    st.write("### 🤖 AI Output")
    output = process_text(st.session_state.transcript)
    st.markdown(output)
