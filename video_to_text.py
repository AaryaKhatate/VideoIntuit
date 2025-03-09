import subprocess
import whisper
import torch
import os
import io
import numpy as np
import soundfile as sf
import yt_dlp
import time
import nltk
import re

# Download nltk resources (if not already downloaded)
nltk.download('punkt', quiet=True)

# Path to FFmpeg binary (replace with your FFmpeg path)
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
        return audio_data.astype(np.float32) / 32768.0  # Normalize
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
                return audio_data.astype(np.float32) / 32768.0
            else:
                print(" No audio found in the video.")
                return None
    except Exception as e:
        print(f" Error extracting audio from video: {e}")
        return None

def transcribe_audio(audio_data):
    """Transcribes audio using Whisper AI with CUDA and preprocesses the transcript."""
    print(" Transcribing with Whisper (GPU)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device}")

    model = whisper.load_model("medium").to(device)

    print(" Detecting language and transcribing...")

    result = model.transcribe(audio_data, language=None)

    detected_language = result.get("language", "Unknown")
    print(f" Detected Language: {detected_language.upper()}")

    transcript = result["text"]

    # Preprocessing
    transcript = transcript.lower()
    transcript = re.sub(r'\s+', ' ', transcript).strip()

    # Sentence segmentation
    sentences = nltk.sent_tokenize(transcript)

    # Basic punctuation and capitalization
    processed_sentences = []
    for sentence in sentences:
        if sentence:
            processed_sentence = sentence.capitalize() + "."
            processed_sentences.append(processed_sentence)

    processed_transcript = " ".join(processed_sentences)

    print("\n **Processed Transcript:**\n", processed_transcript)
    return processed_transcript

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
        transcribe_audio(audio_data)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()