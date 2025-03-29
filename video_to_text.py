import time
import os
import subprocess
import nltk
import torch
import yt_dlp
import ollama
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
        return output_file  
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_from_video_url(video_url, output_file="temp_audio.wav"):
    """Downloads audio from YouTube or online video URL and saves it."""
    print("Extracting audio from YouTube video...")
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
        return "temp_audio.wav"  
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using Faster Whisper with automatic language detection."""
    print("Transcribing with Faster Whisper...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("large", device=device, compute_type="float16" if device == "cuda" else "float32")

    print("Detecting language and transcribing...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    detected_language = getattr(info, "language", "en")
    print(f"Detected Language: {detected_language.upper()}")

    model_size = "medium" if detected_language == "en" else "large"
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "float32")

    print(f"Using {model_size} model for transcription...")
    segments, _ = model.transcribe(audio_path, beam_size=5)

    raw_transcript = " ".join([segment.text for segment in segments])
    print("\n**Raw Transcript:**\n", raw_transcript)

    sentences = nltk.sent_tokenize(raw_transcript)
    processed_transcript = " ".join(sentence.capitalize() for sentence in sentences)

    print("\n**Processed Transcript:**\n", processed_transcript)
    
    return processed_transcript

def answer_question(conversation_history):
    """Uses Ollama (LLaMA 3.2 or Mistral) to generate answers while retaining context."""
    model_name = "llama3.2"
    
    while True:
        question = input("\nEnter another question based on the transcript (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        
        conversation_history.append({"role": "user", "content": question})

        try:
            stream = ollama.chat(
                model=model_name,
                messages=conversation_history,
                stream=True,
            )
            print("\n**Ollama QnA Answer:**\n")
            
            response_content = ""
            for part in stream:
                response_content += part['message']['content']
                print(part['message']['content'], end='', flush=True)
            print("\n")

            conversation_history.append({"role": "assistant", "content": response_content})
        
        except ollama.ResponseError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    start_time = time.time()
    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()
    first_question = input("\nEnter your question based on the transcript: ").strip()
    
    if input_path.startswith("http"):
        audio_path = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_path = extract_audio(input_path)
    else:
        print("Error: Invalid input. Enter a valid URL or video file path.")
        return
    
    if not audio_path:
        return
    
    transcript = transcribe_audio(audio_path)
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    conversation_history = [{"role": "system", "content": f"This is the transcript of the video. Answer the question based on this transcript.\nTranscript of the video: {transcript}"}]
    
    conversation_history.append({"role": "user", "content": first_question})
    
    try:
        stream = ollama.chat(
            model="llama3.2",
            messages=conversation_history,
            stream=True,
        )
        print("\n**Ollama QnA Answer:**\n")
        
        response_content = ""
        for part in stream:
            response_content += part['message']['content']
            print(part['message']['content'], end='', flush=True)
        print("\n")

        conversation_history.append({"role": "assistant", "content": response_content})
    
    except ollama.ResponseError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    answer_question(conversation_history)
    
    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
