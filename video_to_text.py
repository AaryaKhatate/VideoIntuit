import time
import os
import subprocess
import nltk
import torch
import yt_dlp
import ollama
import re
from faster_whisper import WhisperModel

# Download nltk resources
nltk.download('punkt', quiet=True)

# Path to FFmpeg binary (Update as needed)
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"

def extract_audio(video_source, output_file="temp_audio.wav"):
    """Extracts audio from a video file and saves it as a WAV file."""
    print("Extracting audio from video...")

    command = [
        FFMPEG_PATH, "-i", video_source, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", output_file
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_file  # Return the saved audio file path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_from_video_url(video_url, output_file="temp_audio.wav"):
    """Downloads audio from YouTube or online video URL and saves it."""
    print("Downloading video audio...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'outtmpl': "temp_audio.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return "temp_audio.wav"  # Return saved audio file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using Faster Whisper with automatic language detection."""
    print("Transcribing with Faster Whisper...")

    # Use large model for multilingual, medium for English
    model_large = WhisperModel("large", device="cuda", compute_type="float16")
    
    print("Detecting language and transcribing...")
    segments, info = model_large.transcribe(audio_path, beam_size=5)

    # Get detected language
    detected_language = info.language
    print(f"Detected Language: {detected_language.upper()}")

    # Choose optimized model
    model_size = "medium" if detected_language == "en" else "large"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"Using {model_size} model for transcription...")
    segments, _ = model.transcribe(audio_path, beam_size=5)

    # Generate transcript
    raw_transcript = " ".join([segment.text for segment in segments])

    print("\n**Raw Transcript:**\n", raw_transcript)

    # Sentence segmentation
    sentences = nltk.sent_tokenize(raw_transcript)

    # Proper capitalization & punctuation
    processed_transcript = " ".join(sentence.capitalize() for sentence in sentences)

    print("\n**Processed Transcript:**\n", processed_transcript)
    
    return processed_transcript

def answer_question(transcript, question):
    """Uses Ollama (LLaMA 3.2 or Mistral) to generate a streaming answer."""
    model_name = "llama3.2"  # Change to "mistral" if needed

    prompt = f"Answer in detail based on the provided context which is a transcription of a video.\nQuestion: {question}\n\nContext: {transcript}\n\n"

    try:
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        print("\n**Ollama QnA Answer:**\n")
        for part in stream:
            print(part['message']['content'], end='', flush=True)
        print("\n")

    except ollama.ResponseError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    start_time = time.time()

    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()
    question = input("\nEnter your question based on the transcript: ")

    if input_path.startswith("http"):
        audio_path = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_path = extract_audio(input_path)
    else:
        print("Error: Invalid input. Enter a valid URL or video file path.")
        return

    if audio_path:
        transcript = transcribe_audio(audio_path)

        # Delete the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Directly call Ollama for QnA
        answer_question(transcript, question)

    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
