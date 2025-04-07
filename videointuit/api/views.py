# api/views.py
from django.core.cache import cache # Using Django's cache framework
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt # Use with caution
from django.conf import settings # For potential future settings

import time
import logging
import os
import re
import subprocess
import tempfile # Use for temporary files
import json
import nltk
import torch
import yt_dlp
import ollama
from faster_whisper import WhisperModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Model Loading (Load Once at Startup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
WHISPER_MODELS = {}

try:
    # *** CHANGE: Load the "medium" Whisper model ***
    WHISPER_MODEL_NAME = "medium" # Specify desired model size
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' (Device: {DEVICE}, Compute: {COMPUTE_TYPE})...")
    WHISPER_MODELS[WHISPER_MODEL_NAME] = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Whisper models: {e}", exc_info=True)

# Download necessary NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
    logger.info("'punkt' downloaded.")

# --- Configuration ---
FFMPEG_COMMAND = 'ffmpeg' # Assumes ffmpeg is in PATH
CACHE_TIMEOUT = getattr(settings, 'CHAT_CACHE_TIMEOUT', 3600) # Default to 1 hour
OLLAMA_MODEL = "llama3.2" # *** CHANGE: Specify desired Ollama model ***


# --- Helper Functions ---

def extract_audio(video_source_path, target_audio_path):
    """Extracts audio from a local video file using FFmpeg."""
    logger.info(f"Extracting audio from '{video_source_path}' to '{target_audio_path}'...")
    command = [
        FFMPEG_COMMAND,
        "-i", video_source_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", # Overwrite output file if it exists
        target_audio_path
    ]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        logger.info("Audio extraction successful.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error extracting audio: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"FFmpeg command ('{FFMPEG_COMMAND}') not found. Ensure FFmpeg is installed and in the system PATH.")
        return False

def extract_audio_from_video_url(video_url, target_audio_path):
    """Downloads and extracts audio from a video URL using yt-dlp."""
    logger.info(f"Attempting to extract audio from URL: {video_url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': target_audio_path.replace('.wav', '.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        # 'ffmpeg_location': '/path/to/ffmpeg' # Optional: if not in PATH
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if os.path.exists(target_audio_path):
            logger.info("Audio download and extraction via yt-dlp successful.")
            return True
        else:
             # Clean up potentially downloaded intermediate files if wav wasn't created
            base_name = os.path.basename(target_audio_path).replace('.wav','')
            possible_files = [f for f in os.listdir('.') if f.startswith(base_name) and not f.endswith('.wav')]
            for f in possible_files:
                try:
                    os.remove(f)
                    logger.warning(f"Removed intermediate download file: {f}")
                except OSError: pass
            logger.error("yt-dlp post-processing failed to create WAV file.")
            return False
    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during audio download: {e}", exc_info=True)
        return False

def transcribe_audio(audio_path):
    """Transcribes audio using the pre-loaded 'medium' Faster Whisper model."""
    # *** CHANGE: Directly use the pre-loaded medium model ***
    model_key = WHISPER_MODEL_NAME # Use the globally defined model name ('medium')

    if model_key not in WHISPER_MODELS:
         logger.error(f"Whisper model '{model_key}' not loaded. Cannot transcribe.")
         return None, f"Whisper model '{model_key}' unavailable."

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found for transcription: {audio_path}")
        return None, "Audio file missing."

    logger.info(f"Starting transcription using '{model_key}' model for: {audio_path}")
    start_time = time.time()

    try:
        model = WHISPER_MODELS[model_key] # Get the pre-loaded model instance

        # Perform transcription
        segments, info = model.transcribe(audio_path, beam_size=5)

        detected_language = info.language
        lang_probability = info.language_probability
        logger.info(f"Detected language: {detected_language} (Confidence: {lang_probability:.2f})")
        logger.info(f"Transcription duration: {info.duration:.2f}s")

        # Process segments into a readable transcript
        raw_transcript = " ".join([segment.text.strip() for segment in segments])
        sentences = nltk.sent_tokenize(raw_transcript)
        processed_transcript = " ".join(sentence.capitalize() for sentence in sentences)

        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Processed Transcript: {processed_transcript[:500]}...") # Log first part

        return processed_transcript, None # Return transcript and no error
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None, f"Transcription failed: {e}"


# --- Django Views ---

def index(request):
    """Renders the main chat page."""
    return render(request, 'index.html')


@csrf_exempt
def upload_video(request):
    """Handles video file/URL upload, transcription, and stores transcript in cache."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    video_source = None
    source_type = None
    uploaded_file_path = None
    temp_audio_path = None

    try:
        # Determine input type
        if request.FILES.get('videoFile'):
            video_file = request.FILES['videoFile']
            source_type = 'file'
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
                for chunk in video_file.chunks():
                    tmp_video.write(chunk)
                uploaded_file_path = tmp_video.name
            video_source = uploaded_file_path
            logger.info(f"Received uploaded file: {video_file.name}, saved to {uploaded_file_path}")
        elif request.body:
             try:
                 data = json.loads(request.body)
                 if data.get('videoUrl'):
                     video_source = data['videoUrl']
                     source_type = 'url'
                     logger.info(f"Received video URL: {video_source}")
                 else: return JsonResponse({'error': 'Missing videoUrl in JSON payload'}, status=400)
             except json.JSONDecodeError: return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        else: return JsonResponse({'error': 'No video file or URL provided'}, status=400)

        if not video_source: return JsonResponse({'error': 'Failed to identify video source'}, status=400)

        # Audio Extraction
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_f:
            temp_audio_path = tmp_audio_f.name
        extraction_success = False
        if source_type == 'file': extraction_success = extract_audio(video_source, temp_audio_path)
        elif source_type == 'url': extraction_success = extract_audio_from_video_url(video_source, temp_audio_path)
        if not extraction_success or not os.path.exists(temp_audio_path):
             return JsonResponse({'error': 'Audio extraction failed'}, status=500)

        # Transcription (now uses 'medium' model via transcribe_audio)
        transcript, trans_error = transcribe_audio(temp_audio_path)
        if trans_error: return JsonResponse({'error': trans_error}, status=500)

        # Store Transcript in Cache
        if not request.session.session_key: request.session.create()
        session_key = request.session.session_key
        transcript_key = f"transcript_{session_key}"
        cache.set(transcript_key, transcript, timeout=CACHE_TIMEOUT)
        logger.info(f"Transcript stored in cache for session {session_key}")

        return JsonResponse({'message': 'Video processed successfully. Transcript ready.'})

    except Exception as e:
        logger.error(f"Unexpected error in upload_video: {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)
    finally:
        # Cleanup Temporary Files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path); logger.info(f"Cleaned up temp audio file: {temp_audio_path}")
            except OSError as e: logger.warning(f"Could not remove temp audio file {temp_audio_path}: {e}")
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try: os.remove(uploaded_file_path); logger.info(f"Cleaned up temp video file: {uploaded_file_path}")
            except OSError as e: logger.warning(f"Could not remove temp video file {uploaded_file_path}: {e}")


@csrf_exempt
def ask_question(request):
    """Handles user questions, retrieves context, interacts with Ollama (llama3.2), and updates history."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get('question')
        if not question: return JsonResponse({'error': 'No question provided'}, status=400)

        if not request.session.session_key: request.session.create()
        session_key = request.session.session_key

        # Retrieve Context from Cache
        history_key = f"conversation_history_{session_key}"
        transcript_key = f"transcript_{session_key}"
        conversation_history = cache.get(history_key, [])
        transcript = cache.get(transcript_key)

        # Prepare Ollama Payload
        is_new_conversation = not conversation_history
        if is_new_conversation:
            if transcript:
                system_prompt = (
                    f"You are an AI assistant. A video transcript has been provided. "
                    f"Your primary task is to answer the user's questions based *ONLY* on the information contained within the following transcript. "
                    f"Do not use any prior knowledge or external information unless explicitly permitted. "
                    f"If the answer to a question cannot be found in the transcript, you MUST clearly state that the information is not available in the provided text "
                    f"AND THEN ask the user if they would like you to answer using your general knowledge. Do not answer from general knowledge without permission. "
                    f"Transcript:\n---\n{transcript}\n---"
                )
                conversation_history.append({"role": "system", "content": system_prompt})
                logger.info(f"Added transcript-based system prompt to history for session {session_key}")
            else:
                conversation_history.append({"role": "system", "content": "You are a helpful AI assistant."})
                logger.warning(f"No transcript found in cache for session {session_key}. Starting generic conversation.")
        elif not transcript and any(msg.get('role') == 'system' and 'Transcript:' in msg.get('content','') for msg in conversation_history):
             logger.warning(f"Transcript missing from cache mid-conversation for session {session_key}, but history implies it existed.")

        # Add Current User Question
        conversation_history.append({"role": "user", "content": question})
        logger.info(f"Sending request to Ollama model '{OLLAMA_MODEL}' for session {session_key}. History length: {len(conversation_history)}")

        # Interact with Ollama
        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL, 
                messages=conversation_history,
                stream=True
            )
            response_content = ""
            for chunk in stream:
                message_chunk = chunk.get('message', {})
                content_chunk = message_chunk.get('content', '')
                response_content += content_chunk
            if not response_content:
                 logger.warning(f"Ollama returned an empty response for session {session_key}")
                 response_content = "I apologize, I couldn't generate a response for that."

        except ollama.ResponseError as e:
            # Check specifically for model not found error
            if e.status_code == 404:
                 logger.error(f"Ollama model '{OLLAMA_MODEL}' not found for session {session_key}. Error: {e.error}. Please ensure the model is pulled using 'ollama pull {OLLAMA_MODEL}'.")
                 return JsonResponse({'error': f"The AI model '{OLLAMA_MODEL}' was not found on the server. Please contact the administrator."}, status=500)
            else:
                 logger.error(f"Ollama API error for session {session_key}: {e.status_code} - {e.error}")
                 return JsonResponse({'error': f'Ollama error: {e.error}'}, status=500)
        except Exception as e:
             logger.error(f"Error communicating with Ollama for session {session_key}: {e}", exc_info=True)
             return JsonResponse({'error': 'Failed to get response from AI model.'}, status=500)

        # Add Assistant's Response to History
        conversation_history.append({"role": "assistant", "content": response_content})

        # Update Cache
        cache.set(history_key, conversation_history, timeout=CACHE_TIMEOUT)
        if transcript: cache.touch(transcript_key, timeout=CACHE_TIMEOUT) # Keep transcript cache alive
        logger.info(f"Received Ollama response and updated cache for session {session_key}")

        return JsonResponse({'answer': response_content})

    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON body for session {request.session.session_key or 'Unknown'}")
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in ask_question (Session: {request.session.session_key or 'Unknown'}): {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)