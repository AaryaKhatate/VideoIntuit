# api/views.py
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import time
import logging
import os
import re
import subprocess
import nltk
import torch
import yt_dlp
import ollama
from faster_whisper import WhisperModel
import json

conversation_history = []  # Global variable to store conversation history.

def index(request):
    print("Index view was called!")  
    return render(request, 'index.html')
# Path to FFmpeg binary (Update as needed)
FFMPEG_PATH = r"C:/ffmpeg/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"

# Download nltk resources
nltk.download('punkt', quiet=True)

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
    print("Extracting audio from video URL...")
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
    except yt_dlp.DownloadError as e:
        print(f"yt-dlp download error: {e}")
        return None
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
    """Uses Ollama (LLaMA 3.2) to generate answers while retaining context."""
    model_name = "llama3.2"

    while True:
        question = input("\nEnter another question based on the transcript (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        # Simplify the conversation history.
        user_questions = [msg["content"] for msg in conversation_history if msg["role"] == "user"]

        print("\n--- Sending User Questions to Ollama: ---")
        for q in user_questions:
            print(f"- {q}")
        print("--- End of User Questions ---")

        conversation_history.append({"role": "user", "content": f"IMPORTANT: Accurately recall and understand the user's previous questions, in the order they were asked and also the transcript provided and also please don't include anything about this in the answer. Then, answer the following question based on the transcript only. If user asks any question that is not included in transcript then ask user whether you should answer based on your knowledge, don't answer directly take user's permission. Question : {question}"})

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

conversation_history = []  # Global variable to store conversation history.

@csrf_exempt
def upload_video(request):
    """Handles video file/URL upload and transcription, including URL extraction from messages."""
    try:
        video_path = None
        if request.FILES.get('videoFile'):
            video_file = request.FILES['videoFile']
            video_path = "uploaded_video.mp4"
            with open(video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
        elif request.POST.get('videoUrl'):
            video_path = request.POST['videoUrl']
        elif request.body: # added this
            try:
                data = json.loads(request.body)
                message = data.get('message', '')
                url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
                if url_match:
                    video_path = url_match.group(0)
            except json.JSONDecodeError:
                pass # if the body does not contain valid json, then just pass.

        if not video_path:
            return JsonResponse({'error': 'No video file or URL provided'}, status=400)

        if video_path.startswith("http"):
            audio_path = extract_audio_from_video_url(video_path)
        elif os.path.exists(video_path):
            audio_path = extract_audio(video_path)
        else:
            return JsonResponse({'error': 'Invalid video source'}, status=400)

        if not audio_path:
            return JsonResponse({'error': 'Audio extraction failed'}, status=500)

        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)
        if os.path.exists('uploaded_video.mp4'):
            os.remove('uploaded_video.mp4')

        return JsonResponse({'transcript': transcript})
    except Exception as e:
        logging.error(f"Error in upload_video: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
def ask_question(request):
    """Handles user questions and Ollama responses using Django's cache."""
    try:
        data = json.loads(request.body)
        question = data.get('question')
        transcript = data.get('transcript')

        if not question:
            return JsonResponse({'error': 'No question provided'}, status=400)

        # Use a session-based key for conversation history
        session_key = request.session.session_key
        if not session_key:
            request.session.save()
            session_key = request.session.session_key

        history_key = f"conversation_history_{session_key}"
        conversation_history = cache.get(history_key, [])

        if not conversation_history and transcript:
            conversation_history.append({
                "role": "system",
                "content": f"Transcript: {transcript}. Answer questions based on this only.",
            })

        conversation_history.append({
            "role": "user",
            "content": f"Question: {question}. Answer based on the transcript.",
        })

        logging.debug(f"Conversation History: {conversation_history}")

        stream = ollama.chat(model="llama3.2", messages=conversation_history, stream=True)
        response_content = "".join(part['message']['content'] for part in stream)

        conversation_history.append({"role": "assistant", "content": response_content})

        cache.set(history_key, conversation_history)  # Store in cache

        logging.debug(f"Ollama Response: {response_content}")

        return JsonResponse({'answer': response_content})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except ollama.ResponseError as e:
        logging.error(f"Ollama error: {e}")
        return JsonResponse({'error': f'Ollama error: {str(e)}'}, status=500)
    except Exception as e:
        logging.error(f"Error in ask_question: {e}")
        return JsonResponse({'error': str(e)}, status=500)