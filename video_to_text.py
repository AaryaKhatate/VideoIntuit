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

<<<<<<< HEAD
# Download nltk resources (if not already downloaded)
nltk.download('punkt', quiet=True)

# Path to FFmpeg binary (replace with your FFmpeg path)
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
=======
# Path to FFmpeg binary
FFMPEG_PATH = r"C:\Users\Yashraj Patil\Downloads\ffmpeg-2025-01-30-git-1911a6ec26-full_build\ffmpeg-2025-01-30-git-1911a6ec26-full_build\bin\ffmpeg.exe"
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92

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
<<<<<<< HEAD
        'outtmpl': '-',
=======
        'outtmpl': output_audio_path,  # Save as temp file
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
<<<<<<< HEAD
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
=======
            ydl.download([video_url])

        if os.path.exists(output_audio_path):
            # Extract the downloaded audio file
            audio_data = extract_audio(output_audio_path)
            os.remove(output_audio_path)  # Remove temp file after use
            return audio_data
        else:
            print("Error: Audio file was not downloaded.")
            return None
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def transcribe_audio(audio_data):
<<<<<<< HEAD
    """Transcribes audio using Whisper AI with CUDA and preprocesses the transcript."""
    print(" Transcribing with Whisper (GPU)...")
=======
    """Transcribes audio using Whisper AI with CUDA (NVIDIA GPU)."""
    print("Transcribing with Whisper (GPU)...")
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model("medium").to(device)

    print("Detecting language and transcribing...")

    result = model.transcribe(audio_data, language=None)

    detected_language = result.get("language", "Unknown")
    print(f"Detected Language: {detected_language.upper()}")

<<<<<<< HEAD
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
=======
    print("\n**Transcript:**\n", result["text"])
    return result["text"]
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92

def main():
    start_time = time.time()

    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()

    if input_path.startswith("http"):
        audio_data = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_data = extract_audio(input_path)
    else:
        print("Error: Invalid input. Enter a valid URL or video file path.")
        return

    if audio_data is not None:
        transcribe_audio(audio_data)

<<<<<<< HEAD
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n Total time taken: {total_time:.2f} seconds")
=======
    end_time = time.time()  # End the timer
    total_time = end_time - start_time  # Calculate total time
    print(f"\nTotal time taken: {total_time:.2f} seconds")
>>>>>>> 4bd0b7f7dce15c852f70ceaffd82c6ca20434d92

if __name__ == "__main__":
    main()