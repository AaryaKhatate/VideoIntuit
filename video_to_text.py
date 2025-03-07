import subprocess
import whisper
import torch
import os
import io
import numpy as np
import soundfile as sf
import yt_dlp  # Import yt-dlp Python package
import time  # For timing the process

# Path to FFmpeg binary
FFMPEG_PATH = r"C:\Users\Yashraj Patil\Downloads\ffmpeg-2025-01-30-git-1911a6ec26-full_build\ffmpeg-2025-01-30-git-1911a6ec26-full_build\bin\ffmpeg.exe"

def extract_audio(video_path):
    """Extracts audio from a local video file using FFmpeg and returns it as a NumPy array."""
    print("Extracting audio from video...")

    command = [
        FFMPEG_PATH, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"
    ]

    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_data, samplerate = sf.read(io.BytesIO(process.stdout), dtype='int16')
        return audio_data.astype(np.float32) / 32768.0  # Normalize
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_from_video_url(video_url):
    """Extracts audio from a YouTube video using yt-dlp and FFmpeg."""
    print("Fetching video audio stream...")

    output_audio_path = "temp_audio.m4a"  # Temporary audio file

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_audio_path,  # Save as temp file
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(output_audio_path):
            # Extract the downloaded audio file
            audio_data = extract_audio(output_audio_path)
            os.remove(output_audio_path)  # Remove temp file after use
            return audio_data
        else:
            print("Error: Audio file was not downloaded.")
            return None
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def transcribe_audio(audio_data):
    """Transcribes audio using Whisper AI with CUDA (NVIDIA GPU)."""
    print("Transcribing with Whisper (GPU)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model("medium").to(device)  # Use "small", "medium", or "large"

    print("Detecting language and transcribing...")

    # Transcribe with auto-detect language
    result = model.transcribe(audio_data, language=None)  # Auto-detect language

    detected_language = result.get("language", "Unknown")
    print(f"Detected Language: {detected_language.upper()}")

    print("\n**Transcript:**\n", result["text"])
    return result["text"]

def main():
    start_time = time.time()  # Start the timer

    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()

    # Handle YouTube URL, Vimeo URL, or local video path
    if input_path.startswith("http"):
        audio_data = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_data = extract_audio(input_path)
    else:
        print("Error: Invalid input. Enter a valid URL or video file path.")
        return

    # Proceed with transcription if audio is successfully extracted
    if audio_data is not None:
        transcribe_audio(audio_data)

    end_time = time.time()  # End the timer
    total_time = end_time - start_time  # Calculate total time
    print(f"\nTotal time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
