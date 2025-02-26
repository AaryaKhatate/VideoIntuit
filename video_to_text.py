import subprocess
import whisper
import torch
import os
import io
import numpy as np
import soundfile as sf
import yt_dlp
import time
import librosa
from transformers import pipeline
from textblob import TextBlob
import re

# Path to FFmpeg binary
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"

def extract_audio(video_path):
    """Extracts audio from a local video file using FFmpeg and returns it as a NumPy array."""
    print(" Extracting audio from video...")

    command = [
        FFMPEG_PATH, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"
    ]

    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_data, samplerate = sf.read(io.BytesIO(process.stdout), dtype='int16')
        return preprocess_audio(audio_data.astype(np.float32) / 32768.0)  # Normalize & preprocess
    except subprocess.CalledProcessError as e:
        print(f" Error extracting audio: {e}")
        return None

def extract_audio_from_video_url(video_url):
    """Extracts audio from various websites using yt-dlp."""
    print(" Fetching video audio stream...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '-',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            audio_formats = [f for f in info_dict['formats'] if f.get('acodec') != 'none']
            if audio_formats:
                audio_url = audio_formats[0]['url']
                command = [
                    FFMPEG_PATH, "-i", audio_url, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"
                ]
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                audio_data, samplerate = sf.read(io.BytesIO(process.stdout), dtype='int16')
                return preprocess_audio(audio_data.astype(np.float32) / 32768.0)
            else:
                print(" No audio found in the video.")
                return None
    except Exception as e:
        print(f" Error extracting audio from video: {e}")
        return None

def preprocess_audio(audio_data, sample_rate=16000):
    """Enhances audio quality by reducing noise and normalizing volume."""
    audio_data = librosa.to_mono(audio_data)  # Convert to mono if necessary
    audio_data = librosa.effects.preemphasis(audio_data)  # Apply high-pass filter for noise reduction
    audio_data = audio_data / max(abs(audio_data))  # Normalize volume
    return audio_data

def transcribe_audio(audio_data):
    """Transcribes audio using Whisper AI with enhanced settings."""
    print(" Transcribing with Whisper...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium").to(device)  # Try "large" for better accuracy
    print(f"using device {device}")

    print(" Detecting language and transcribing...")

    result = model.transcribe(
        audio_data,
        language=None,  # Auto-detect language
        fp16=False,  # Full precision for better accuracy
        temperature=0.0,  # More deterministic results
        condition_on_previous_text=False,  # Prevents context bias
        word_timestamps=True,  # Get precise timestamps
    )

    detected_language = result.get("language", "Unknown")
    print(f" Detected Language: {detected_language.upper()}")

    print("\n **Transcript:**\n", result["text"])
    return result["text"]

def clean_transcript(transcript):
    """Post-process transcript: Fix spelling, grammar, and punctuation."""
    
    # Restore punctuation
    punctuator = pipeline("text2text-generation", model="bhdzitao/Punctuator")
    transcript = punctuator(transcript)[0]["generated_text"]
    
    # Fix spelling & grammar
    transcript = str(TextBlob(transcript).correct())

    # Remove filler words
    filler_words = ["um", "uh", "like", "you know", "I mean"]
    pattern = r"\b(" + "|".join(filler_words) + r")\b"
    transcript = re.sub(pattern, "", transcript, flags=re.IGNORECASE)

    return transcript

def main():
    start_time = time.time()

    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()

    if input_path.startswith("http"):
        audio_data = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_data = extract_audio(input_path)
    else:
        print(" Error: Invalid input. Enter a valid URL or video file path.")
        return

    if audio_data is not None:
        transcript = transcribe_audio(audio_data)
        cleaned_transcript = clean_transcript(transcript)
        print("\n **Cleaned Transcript:**\n", cleaned_transcript)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
