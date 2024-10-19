import os
import sounddevice as sd
import numpy as np
import wave
import streamlit as st
from pocketsphinx import Decoder, Config  # Import Config for the updated method
from nltk.corpus import cmudict
import librosa  # For audio preprocessing

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

# Set parameters for recording
SAMPLE_RATE = 16000
FILENAME = "live_recording.wav"
MAX_DURATION = 10

# Function to preprocess audio (e.g., noise reduction)
def preprocess_audio(audio_file):
    """Apply noise reduction or any other preprocessing steps."""
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    # Apply any preprocessing steps, such as noise reduction, filtering, etc.
    return y

# Function to record audio
def record_audio(max_duration, sample_rate, filename):
    """Records audio from the microphone and saves it as a WAV file."""
    try:
        st.info("Recording started...")
        progress_bar = st.progress(0)  # Progress bar to indicate recording status
        audio = sd.rec(int(max_duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        
        # Simulate recording duration with progress bar
        for i in range(max_duration):
            sd.sleep(1000)  # Simulate recording time (1 second at a time)
            progress_bar.progress((i + 1) / max_duration)
        
        sd.wait()  # Wait for the recording to finish
        st.success("Recording completed.")
        
        # Save the recorded audio as a WAV file
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # Sample width in bytes (16-bit audio)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        st.success(f"Audio saved to {filename}")
    except Exception as e:
        st.error(f"Error recording audio: {e}")

# Function to transcribe phonemes using PocketSphinx
def recognize_phonemes_with_pocketsphinx(audio_file):
    """Recognizes phonemes from the audio using PocketSphinx."""
    try:
        st.info("Recognizing phonemes with PocketSphinx...")
        
        # Preprocess the audio
        preprocess_audio(audio_file)
        
        # Dynamic user input for model paths
        acoustic_model_path = st.text_input("Enter acoustic model path:", value=r'C:\Users\Dell\Downloads\cmusphinx-en-us-8khz-5.2\cmusphinx-en-us-8khz-5.2')
        language_model_path = st.text_input("Enter language model path:", value=r'C:\Users\Dell\Downloads\en-us.lm.bin')
        dictionary_path = st.text_input("Enter dictionary path:", value=r'C:\Users\Dell\Downloads\cmudict-en-us.dict')
        
        # Set up decoder with phoneme configuration
        config = Config()  # Create a new Config instance
        config.set_string('-hmm', acoustic_model_path)  # Acoustic model path
        config.set_string('-lm', language_model_path)   # Language model path
        config.set_string('-dict', dictionary_path)     # Dictionary path

        decoder = Decoder(config)

        # Open the audio file and start decoding phonemes
        with open(audio_file, 'rb') as f:
            decoder.start_utt()
            while True:
                buf = f.read(1024)
                if buf:
                    decoder.process_raw(buf, False, False)
                else:
                    break
            decoder.end_utt()

        # Extract phonemes
        phonemes = [seg.word for seg in decoder.seg()]
        phoneme_sequence = " ".join(phonemes)
        st.subheader("Recognized Phonemes:")
        st.text(phoneme_sequence)
        return phoneme_sequence
    except Exception as e:
        st.error(f"Error recognizing phonemes: {e}")
        return None

# Function to map phonemes back to words
def map_phonemes_to_words(phoneme_sequence):
    """Maps recognized phonemes to words using CMU Pronouncing Dictionary."""
    try:
        st.info("Mapping phonemes back to words...")
        words = []
        for word, phoneme_list in d.items():
            # Check if any of the CMU dictionary phoneme lists match the recognized phoneme sequence
            for phoneme_set in phoneme_list:
                phoneme_str = " ".join(phoneme_set)
                if phoneme_str in phoneme_sequence:
                    words.append(word)
        return " ".join(words)
    except Exception as e:
        st.error(f"Error mapping phonemes to words: {e}")
        return ""

# Main Streamlit app
def main():
    st.title("Phoneme-First Audio Transcription Without Auto-Correction")
    st.write("This app records live audio, recognizes phonemes first using PocketSphinx, and maps them back to words without auto-correction.")

    # Initialize phoneme_sequence at the start
    phoneme_sequence = None

    # Provide the option to upload a pre-recorded audio file
    uploaded_file = st.file_uploader("Upload a pre-recorded audio file", type=["wav"])
    if uploaded_file is not None:
        # Save the uploaded file to disk and process it
        with open(FILENAME, 'wb') as f:
            f.write(uploaded_file.read())
        phoneme_sequence = recognize_phonemes_with_pocketsphinx(FILENAME)

        if phoneme_sequence:
            words = map_phonemes_to_words(phoneme_sequence)
            st.subheader("Mapped Words (No Auto-Correction):")
            st.text(words)
    
    # Button to record audio and automatically transcribe & extract phonemes
    if st.button("Record Audio"):
        # Step 1: Record the audio live
        record_audio(MAX_DURATION, SAMPLE_RATE, FILENAME)
        
        # Step 2: Recognize phonemes using PocketSphinx
        if os.path.exists(FILENAME):
            phoneme_sequence = recognize_phonemes_with_pocketsphinx(FILENAME)

            # Step 3: Map phonemes to words
            if phoneme_sequence:
                words = map_phonemes_to_words(phoneme_sequence)

                st.subheader("Mapped Words (No Auto-Correction):")
                st.text(words)
    
    # Option to download the recognized phoneme sequence
    if phoneme_sequence:
        st.download_button("Download Phoneme Output", phoneme_sequence)

if __name__ == "__main__":
    main()
