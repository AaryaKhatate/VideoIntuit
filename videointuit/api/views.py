# api/views.py
from django.core.cache import cache # Using Django's cache framework
from django.http import StreamingHttpResponse, HttpResponseNotAllowed, JsonResponse # Correct imports
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
    WHISPER_MODEL_NAME = getattr(settings, 'WHISPER_MODEL_NAME', "medium") # Get from settings or default
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' (Device: {DEVICE}, Compute: {COMPUTE_TYPE})...")
    WHISPER_MODELS[WHISPER_MODEL_NAME] = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Whisper models: {e}", exc_info=True)
    # Depending on severity, you might want to raise an exception here to prevent startup
    # raise RuntimeError(f"Failed to load Whisper model: {e}") from e

# Download necessary NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
    logger.info("'punkt' downloaded.")

# --- Configuration ---
FFMPEG_COMMAND = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg') # Assumes ffmpeg is in PATH
CACHE_TIMEOUT = getattr(settings, 'CHAT_CACHE_TIMEOUT', 3600) # Default to 1 hour
OLLAMA_MODEL = getattr(settings, 'OLLAMA_MODEL', 'llama3.2') # Get model from settings or default


# --- Helper Functions ---

def extract_audio(video_source_path, target_audio_path):
    """Extracts audio from a local video file using FFmpeg."""
    logger.info(f"Extracting audio from '{video_source_path}' to '{target_audio_path}'...")
    command = [
        FFMPEG_COMMAND,
        "-i", video_source_path,
        "-vn", # No video output
        "-acodec", "pcm_s16le", # Standard WAV codec
        "-ar", "16000", # Sample rate for Whisper
        "-ac", "1", # Mono channel
        "-y", # Overwrite output file if it exists
        target_audio_path
    ]
    try:
        process = subprocess.run(command, capture_output=True, check=True, text=True, encoding='utf-8', errors='replace')
        logger.info("Audio extraction successful.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error extracting audio: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"FFmpeg command ('{FFMPEG_COMMAND}') not found. Ensure FFmpeg is installed and in the system PATH.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during FFmpeg audio extraction: {e}", exc_info=True)
        return False


def extract_audio_from_video_url(video_url, target_audio_path):
    """Downloads and extracts audio from a video URL using yt-dlp."""
    logger.info(f"Attempting to extract audio from URL: {video_url}")
    # Ensure the output template uses the target_audio_path base name but allows yt-dlp to manage extension initially
    base_target = target_audio_path.rsplit('.', 1)[0]
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{base_target}.%(ext)s', # Let yt-dlp determine initial extension
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav', # Convert to wav
            'preferredquality': '192', # Quality setting (doesn't directly apply to wav but often required)
        }],
        'postprocessor_args': { # Correct way to pass args for conversion
             'extractaudio': ['-ar', '16000', '-ac', '1'] # Set sample rate and channels during conversion
        }
        # 'ffmpeg_location': '/path/to/ffmpeg' # Optional: if not in PATH
    }
    downloaded_correctly = False
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        # Check if the final target WAV file exists
        if os.path.exists(target_audio_path):
            downloaded_correctly = True
            logger.info("Audio download and extraction via yt-dlp successful.")
            return True
        else:
            # Check if intermediate file exists (e.g., .webm, .m4a) - yt-dlp might have downloaded but failed conversion
            possible_intermediate = f"{base_target}.{ydl.extract_info(video_url, download=False).get('ext', 'unknown')}"
            if os.path.exists(possible_intermediate):
                 logger.error(f"yt-dlp downloaded '{possible_intermediate}' but failed to convert to WAV at '{target_audio_path}'. Check FFmpeg setup.")
            else:
                 logger.error("yt-dlp failed to download or process the audio.")
            return False
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during audio download/extraction: {e}", exc_info=True)
        return False
    finally:
         # Cleanup: Remove any files matching the base name if the target wav wasn't created
         if not downloaded_correctly:
              try:
                  parent_dir = os.path.dirname(base_target)
                  base_filename = os.path.basename(base_target)
                  for f in os.listdir(parent_dir or '.'):
                        if f.startswith(base_filename):
                            try:
                                full_path = os.path.join(parent_dir, f)
                                os.remove(full_path)
                                logger.warning(f"Cleaned up intermediate/failed download file: {full_path}")
                            except OSError:
                                pass # Ignore errors during cleanup
              except Exception as cleanup_err:
                   logger.warning(f"Error during download cleanup: {cleanup_err}")


def transcribe_audio(audio_path):
    """Transcribes audio using the pre-loaded Faster Whisper model."""
    global WHISPER_MODEL_NAME # Access the global variable
    model_key = WHISPER_MODEL_NAME

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
        # Consider adding 'word_timestamps=True' if you need timing info later
        segments, info = model.transcribe(audio_path, beam_size=5)

        detected_language = info.language
        lang_probability = info.language_probability
        logger.info(f"Detected language: {detected_language} (Confidence: {lang_probability:.2f})")
        logger.info(f"Transcription duration: {info.duration:.2f}s")

        # Process segments into a readable transcript
        # Using generator expression for potentially better memory usage
        raw_transcript = " ".join(segment.text.strip() for segment in segments)

        # Sentence tokenization and capitalization
        try:
             sentences = nltk.sent_tokenize(raw_transcript)
             processed_transcript = " ".join(sentence.capitalize() for sentence in sentences)
        except Exception as nltk_err:
             logger.warning(f"NLTK sentence tokenization failed: {nltk_err}. Using raw transcript.")
             processed_transcript = raw_transcript.capitalize() # Basic fallback


        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Processed Transcript (first 500 chars): {processed_transcript[:500]}...")

        return processed_transcript, None # Return transcript and no error
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None, f"Transcription failed: {e}"


# --- Django Views ---

# View 1: Render the main page
def index(request):
    """Renders the main chat page (index.html)."""
    return render(request, 'index.html')


# View 2: Handle video upload (POST only)
@csrf_exempt
def upload_video(request):
    """
    Handles POST requests for video file/URL upload, extracts audio,
    transcribes it, and stores the transcript in the cache.
    """
    if request.method != 'POST':
        logger.warning(f"Method Not Allowed for upload_video: {request.method}")
        return HttpResponseNotAllowed(['POST'])

    video_source = None
    source_type = None
    uploaded_file_path = None
    temp_audio_path = None
    initial_question = None # Check if question comes with upload

    try:
        # Handle potential question sent with FormData (e.g., from JS)
        initial_question = request.POST.get('question')

        # Determine input type (File)
        if request.FILES.get('videoFile'):
            video_file = request.FILES['videoFile']
            source_type = 'file'
            # Use tempfile context manager for safer handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
                for chunk in video_file.chunks():
                    tmp_video.write(chunk)
                uploaded_file_path = tmp_video.name # Get path before file closes
            video_source = uploaded_file_path
            logger.info(f"Received uploaded file: {video_file.name}, saved to {uploaded_file_path}")
            if initial_question: logger.info(f"Received initial question with file: {initial_question[:50]}...")

        # Determine input type (URL - Check content type and body)
        elif request.content_type == 'application/json' and request.body:
            try:
                data = json.loads(request.body)
                if data.get('videoUrl'):
                    video_source = data['videoUrl']
                    source_type = 'url'
                    initial_question = data.get('question') # Check for question in JSON
                    logger.info(f"Received video URL: {video_source}")
                    if initial_question: logger.info(f"Received initial question with URL: {initial_question[:50]}...")
                else:
                    logger.warning("upload_video received JSON payload missing 'videoUrl'")
                    return JsonResponse({'error': 'Missing videoUrl in JSON payload'}, status=400)
            except json.JSONDecodeError:
                logger.warning("upload_video received invalid JSON payload")
                return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        else:
            # Neither file nor valid JSON URL detected
            logger.warning("upload_video request with no video file or valid URL payload")
            return JsonResponse({'error': 'No video file or valid URL payload provided'}, status=400)

        # --- Audio Extraction ---
        # Use context manager for temp audio file as well
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_f:
             temp_audio_path = tmp_audio_f.name

        extraction_success = False
        logger.info(f"Attempting audio extraction (Type: {source_type})")
        if source_type == 'file': extraction_success = extract_audio(video_source, temp_audio_path)
        elif source_type == 'url': extraction_success = extract_audio_from_video_url(video_source, temp_audio_path)

        if not extraction_success or not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            logger.error(f"Audio extraction failed or produced empty file for source: {video_source}")
            # Clean up empty/failed audio file before returning
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try: os.remove(temp_audio_path)
                 except OSError: pass
            return JsonResponse({'error': 'Audio extraction failed'}, status=500)

        # --- Transcription ---
        transcript, trans_error = transcribe_audio(temp_audio_path)
        if trans_error:
            # Clean up audio file before returning error
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try: os.remove(temp_audio_path)
                 except OSError: pass
            return JsonResponse({'error': trans_error}, status=500)
        if not transcript:
             logger.error(f"Transcription produced empty result for audio: {temp_audio_path}")
             # Clean up audio file
             if temp_audio_path and os.path.exists(temp_audio_path):
                  try: os.remove(temp_audio_path)
                  except OSError: pass
             return JsonResponse({'error': 'Transcription failed to produce text.'}, status=500)


        # --- Store Transcript in Cache ---
        if not request.session.session_key:
            logger.info("Creating new session for transcript storage.")
            request.session.create()
        session_key = request.session.session_key
        transcript_key = f"transcript_{session_key}"
        cache.set(transcript_key, transcript, timeout=CACHE_TIMEOUT)
        logger.info(f"Transcript stored in cache for session {session_key}")

        # --- Response ---
        # Just confirm processing is done. Frontend handles asking questions separately.
        return JsonResponse({'message': 'Video processed successfully. Transcript ready.'})

    except Exception as e:
        # Catch-all for unexpected errors during the process
        logger.error(f"Unexpected error in upload_video (Source type: {source_type}): {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred during video processing.'}, status=500)

    finally:
        # --- Cleanup Temporary Files ---
        # Ensure cleanup happens even if errors occurred earlier
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temp audio file: {temp_audio_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp audio file {temp_audio_path}: {e}")
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                logger.info(f"Cleaned up temp video file: {uploaded_file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp video file {uploaded_file_path}: {e}")


@csrf_exempt # Keep if needed for POST
def ask_question(request):
    """
    Handles POST requests with a question, retrieves context, gets the FULL
    response from Ollama, updates history, and returns the complete answer.
    """
    # 1. Expect POST method
    if request.method != 'POST':
        logger.warning(f"Method Not Allowed for ask_question: {request.method}")
        return HttpResponseNotAllowed(['POST'])

    # 2. Decode JSON body
    try:
        data = json.loads(request.body)
        question = data.get('question')
        if not question:
            logger.warning("ask_question POST request missing 'question' in JSON body.")
            return JsonResponse({'error': 'No question provided'}, status=400)
    except json.JSONDecodeError:
        logger.warning("ask_question received invalid JSON payload.")
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)

    # 3. Handle session
    if not request.session.session_key:
        # If using POST, we might be able to create a session if one doesn't exist
        logger.info("Creating new session for ask_question POST request.")
        request.session.create()
        # However, this relies on session middleware being set up correctly.
        # If session creation fails or isn't desired here, return an error.
        # If session is strictly required:
        # logger.error("ask_question POST request without a valid session key.")
        # return JsonResponse({'error': 'Session not found. Please ensure cookies are enabled.'}, status=400)

    session_key = request.session.session_key
    logger.info(f"Processing ask_question POST request (Session: {session_key}) for question: {question[:50]}...")

    # 4. Retrieve Context from Cache
    history_key = f"conversation_history_{session_key}"
    transcript_key = f"transcript_{session_key}"
    conversation_history = cache.get(history_key, [])
    transcript = cache.get(transcript_key)

    # 5. Prepare Ollama Payload (Same logic as before)
    is_new_conversation = not conversation_history
    if is_new_conversation:
        if transcript:
            system_prompt = (
                f"You are an AI assistant. A video transcript has been provided. "
                # ... (rest of your system prompt) ...
                f"Transcript:\n---\n{transcript}\n---"
            )
            conversation_history_for_ollama_base = [{"role": "system", "content": system_prompt}]
            logger.info(f"Using transcript-based system prompt for session {session_key}")
        else:
            conversation_history_for_ollama_base = [{"role": "system", "content": "You are a helpful AI assistant."}]
            logger.warning(f"No transcript found for new conversation session {session_key}.")
    else:
        conversation_history_for_ollama_base = conversation_history
        if not transcript and any(msg.get('role') == 'system' and 'Transcript:' in msg.get('content','') for msg in conversation_history):
             logger.warning(f"Transcript missing from cache mid-conversation for session {session_key}.")


    # Add Current User Question for the Ollama call
    conversation_history_for_ollama = conversation_history_for_ollama_base + [{"role": "user", "content": question}]
    logger.info(f"Sending request to Ollama model '{OLLAMA_MODEL}' for session {session_key}. History len: {len(conversation_history_for_ollama)}")

    # 6. Interact with Ollama - Get FULL Response
    full_response_content = ""
    try:
        # Use stream=True and accumulate, or stream=False if the library supports it well
        # Accumulating from stream=True is generally reliable
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=conversation_history_for_ollama,
            stream=True # Or False if preferred and works
        )
        for chunk in stream:
            message_chunk = chunk.get('message', {})
            content_chunk = message_chunk.get('content', '')
            if content_chunk:
                full_response_content += content_chunk

        if not full_response_content:
             logger.warning(f"Ollama returned an empty response stream for session {session_key}")
             full_response_content = "I apologize, I couldn't generate a response for that." # Provide a default

        logger.info(f"Ollama response received for session {session_key}. Length: {len(full_response_content)}")

    except ollama.ResponseError as e:
        error_message = f"Ollama API error: {e.status_code} - {e.error}"
        if e.status_code == 404: error_message = f"The AI model '{OLLAMA_MODEL}' was not found."
        logger.error(f"{error_message} (Session: {session_key})")
        return JsonResponse({'error': error_message}, status=500) # Return error as JSON
    except Exception as e:
        logger.error(f"Error communicating with Ollama for session {session_key}: {e}", exc_info=True)
        return JsonResponse({'error': 'Failed to get response from AI model.'}, status=500) # Return error as JSON

    # 7. Update Cache (using the correct base history)
    if full_response_content and full_response_content != "I apologize, I couldn't generate a response for that.":
        if is_new_conversation:
             base_history_for_update = conversation_history_for_ollama_base
        else:
             base_history_for_update = conversation_history

        updated_history = base_history_for_update + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_response_content}
        ]
        cache.set(history_key, updated_history, timeout=CACHE_TIMEOUT)
        if transcript: cache.touch(transcript_key, timeout=CACHE_TIMEOUT)
        logger.info(f"Updated conversation history in cache for session {session_key}")
    else:
         logger.warning(f"No valid assistant response generated, history not updated for session {session_key}")


    # 8. Return the complete answer
    return JsonResponse({'answer': full_response_content})