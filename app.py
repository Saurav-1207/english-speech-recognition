import google.generativeai as genai
import os
import string
import speech_recognition as sr
from nltk.corpus import cmudict
import streamlit as st
import sounddevice as sd
import numpy as np
import wave

# Configure Google Gemini API
genai.configure(api_key='AIzaSyBndIVJE0ewCiT6d0uaC8iZCxh8luijpds')  # Replace with your actual API key

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

# Set parameters for recording
SAMPLE_RATE = 16000  # Sample rate in Hertz
FILENAME = "live_recording.wav"  # Output file
MAX_DURATION = 10  # Maximum recording duration in seconds

# Function to remove punctuation from transcribed text
def clean_text(text):
    """Removes punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to get phonemes from CMU Pronouncing Dictionary
def get_phonemes_from_text(text):
    """Extracts phonemes for each word in the text using the CMU Pronouncing Dictionary."""
    transcript_words = text.lower().split()
    formatted_phonemes = []

    for word in transcript_words:
        phoneme_list = d.get(word)
        if phoneme_list:
            phoneme_str = " ".join(phoneme_list[0])  # Join phonemes with spaces
            formatted_phonemes.append(f"/{phoneme_str}/")
        else:
            formatted_phonemes.append(f"/UNKNOWN/")

    return " ".join(formatted_phonemes)

# Function to record audio
def record_audio(max_duration, sample_rate, filename):
    """Records audio from the microphone and saves it as a WAV file."""
    st.info("Recording started...")
    audio = sd.rec(int(max_duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    
    # Wait until the recording is finished
    sd.wait()  
    st.success("Recording completed.")

    # Save the recorded audio as a WAV file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes (16-bit audio)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    st.success(f"Audio saved to {filename}")

# Function to upload audio and generate transcription using Gemini
def transcribe_audio_with_gemini(audio_file_path):
    """Uploads the audio file to Gemini and generates a transcription."""
    myfile = genai.upload_file(audio_file_path)
    st.success(f"Uploaded audio file: {myfile}")

    # Create a prompt for transcription
    prompt = "Transcribe the following audio file to text."

    # Send the transcription request to Google Gemini
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content([prompt, myfile])
    
    if response.text:
        st.success("Transcribed Text from Gemini:")
        st.text(response.text)
        return response.text
    else:
        st.error("No transcription was generated.")
        return ""

# Main Streamlit app
def main():
    st.title("Live Audio Transcription and Phoneme Extraction with Google Gemini")
    st.write("This app records live audio, transcribes it using Google Gemini, and extracts phonemes.")

    # Button to record audio and automatically transcribe & extract phonemes
    if st.button("Record Audio"):
        # Step 1: Record the audio live
        record_audio(MAX_DURATION, SAMPLE_RATE, FILENAME)
        
        # Step 2: Transcribe audio using Google Gemini
        if os.path.exists(FILENAME):
            transcribed_text = transcribe_audio_with_gemini(FILENAME)

            if transcribed_text:
                # Step 3: Clean the transcript and extract phonemes
                cleaned_transcript = clean_text(transcribed_text)
                phonemes = get_phonemes_from_text(cleaned_transcript)

                st.subheader("Transcribed Text:")
                st.text(transcribed_text)

                st.subheader("Extracted Phonemes (CMU):")
                st.text(phonemes)

if __name__ == "__main__":
    main()
