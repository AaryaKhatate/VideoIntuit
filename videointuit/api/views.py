# api/views.py
from django.core.cache import cache
from django.http import StreamingHttpResponse, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt # Use with caution, consider CSRF protection
from django.conf import settings

import time
import logging
import os
import re
import subprocess
import tempfile
import json
import nltk
import torch
import yt_dlp
import ollama
# import numpy as np # Commented out as direct use in views is bypassed
# import faiss # Commented out as direct use in views is bypassed
import spacy # Kept for potential future use of chunking
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer # Kept for potential future use of embeddings/RAG
from spellchecker import SpellChecker # For preprocessing

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global Model Loading (Load Once at Startup) ---
# It's crucial these models load successfully for the app to work.
# Consider adding more robust error handling or checks in production.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32" # or "int8_float16" etc.
WHISPER_MODELS = {}
SPACY_NLP = None
EMBEDDING_MODEL = None
SPELL_CHECKER = None
EMBEDDING_DIMENSION = None # Kept for potential future use

# Load Whisper Model
try:
    WHISPER_MODEL_NAME = getattr(settings, 'WHISPER_MODEL_NAME', "medium")
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' (Device: {DEVICE}, Compute: {COMPUTE_TYPE})...")
    # Add other potential options like cpu_threads, num_workers if needed for performance tuning
    WHISPER_MODELS[WHISPER_MODEL_NAME] = WhisperModel(
        WHISPER_MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Whisper model '{WHISPER_MODEL_NAME}': {e}", exc_info=True)
    # Optional: raise RuntimeError("Whisper model failed to load")

# Download/Load spaCy Model (Still potentially useful for future chunking/preprocessing)
try:
    SPACY_MODEL_NAME = getattr(settings, 'SPACY_MODEL_NAME', "en_core_web_sm")
    logger.info(f"Loading spaCy model '{SPACY_MODEL_NAME}'...")
    SPACY_NLP = spacy.load(SPACY_MODEL_NAME)
    logger.info(f"spaCy model '{SPACY_MODEL_NAME}' loaded successfully.")
except OSError:
    logger.warning(f"spaCy model '{SPACY_MODEL_NAME}' not found. Attempting download...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL_NAME], check=True, capture_output=True)
        SPACY_NLP = spacy.load(SPACY_MODEL_NAME)
        logger.info(f"spaCy model '{SPACY_MODEL_NAME}' downloaded and loaded successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to download/load spaCy model '{SPACY_MODEL_NAME}': {e}", exc_info=True)
        # Optional: raise RuntimeError("spaCy model failed to load")
except Exception as e:
     logger.error(f"CRITICAL: Failed to load spaCy model: {e}", exc_info=True)
     # Optional: raise RuntimeError("spaCy model failed to load")

# Load Sentence Transformer Model (Kept for potential future RAG)
try:
    EMBEDDING_MODEL_NAME = getattr(settings, 'EMBEDDING_MODEL_NAME', "all-MiniLM-L6-v2")
    logger.info(f"Loading Sentence Transformer model '{EMBEDDING_MODEL_NAME}'...")
    EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE) # Use configured device
    EMBEDDING_DIMENSION = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    logger.info(f"Sentence Transformer model '{EMBEDDING_MODEL_NAME}' loaded (Dim: {EMBEDDING_DIMENSION}).")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Sentence Transformer model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    # Optional: raise RuntimeError("Embedding model failed to load")

# Initialize Spell Checker
try:
    logger.info("Initializing SpellChecker...")
    SPELL_CHECKER = SpellChecker()
    logger.info("SpellChecker initialized.")
except Exception as e:
    logger.error(f"Failed to initialize SpellChecker: {e}", exc_info=True)
    # Spellchecker might be less critical

# Download necessary NLTK data (only punkt needed now)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    try:
        nltk.download('punkt', quiet=True)
        logger.info("'punkt' downloaded.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt': {e}")


# --- Configuration ---
# Get paths/models from settings or use defaults
FFMPEG_COMMAND = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
CACHE_TIMEOUT = getattr(settings, 'CHAT_CACHE_TIMEOUT', 3600) # 1 hour for transcript/embeddings cache
OLLAMA_MODEL = getattr(settings, 'OLLAMA_MODEL', 'llama3.2') # Updated default
CHUNK_SIZE = getattr(settings, 'TRANSCRIPT_CHUNK_SIZE', 512) # Kept for potential future use
# RAG_TOP_K = getattr(settings, 'RAG_TOP_K', 3) # Commented out as RAG is bypassed
MAX_HISTORY_TURNS = getattr(settings, 'MAX_HISTORY_TURNS', 10)

# --- Helper Functions ---

def extract_audio_from_file(video_source_path, target_audio_path):
    """Extracts audio from a local video file using FFmpeg."""
    logger.info(f"Extracting audio from file '{os.path.basename(video_source_path)}'...")
    command = [
        FFMPEG_COMMAND,
        "-i", video_source_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", # Reduce ffmpeg verbosity in logs
        "-y", # Overwrite output
        target_audio_path
    ]
    try:
        process = subprocess.run(command, capture_output=True, check=True, text=True, encoding='utf-8', errors='replace')
        logger.info("Audio extraction from file successful.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error (File): {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"FFmpeg command ('{FFMPEG_COMMAND}') not found.")
        return False
    except Exception as e:
        logger.error(f"Unexpected FFmpeg Error (File): {e}", exc_info=True)
        return False

def extract_audio_from_video_url(video_url, target_audio_path):
    """Downloads and extracts audio from a video URL using yt-dlp."""
    logger.info(f"Attempting to extract audio from URL: {video_url}")
    base_target = target_audio_path.rsplit('.', 1)[0]
    # Ensure FFMPEG path is correctly passed to yt-dlp if not in system PATH
    ffmpeg_location = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
    ffmpeg_location = ffmpeg_location if ffmpeg_location != 'ffmpeg' else None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{base_target}.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': {
            'extractaudio': ['-ar', '16000', '-ac', '1'] # Ensure Whisper format
        },
        'ffmpeg_location': ffmpeg_location
    }
    downloaded_correctly = False
    try:
        # Suppress yt-dlp INFO logs within this block if desired
        # with yt_dlp.YoutubeDL(ydl_opts) as ydl: ...
        yt_dlp_logger = logging.getLogger('yt_dlp')
        original_level = yt_dlp_logger.getEffectiveLevel()
        yt_dlp_logger.setLevel(logging.WARNING) # Temporarily reduce logging

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        yt_dlp_logger.setLevel(original_level) # Restore logging level

        if os.path.exists(target_audio_path) and os.path.getsize(target_audio_path) > 0:
            downloaded_correctly = True
            logger.info("Audio download and extraction via yt-dlp successful.")
            return True
        else:
            logger.error(f"yt-dlp failed to create target WAV file: {target_audio_path}")
            return False # Let finally block handle cleanup
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        yt_dlp_logger.setLevel(original_level) # Restore on error too
        return False
    except Exception as e:
        logger.error(f"Unexpected yt-dlp Error: {e}", exc_info=True)
        yt_dlp_logger.setLevel(original_level) # Restore on error too
        return False
    finally:
        # Cleanup intermediate files if final WAV wasn't created properly
        if not downloaded_correctly:
            try:
                parent_dir = os.path.dirname(base_target) or '.'
                base_filename = os.path.basename(base_target)
                for f in os.listdir(parent_dir):
                    # Delete files starting with base name but NOT the target wav itself
                    if f.startswith(base_filename) and f != os.path.basename(target_audio_path):
                        try:
                            full_path = os.path.join(parent_dir, f)
                            os.remove(full_path)
                            logger.warning(f"Cleaned up intermediate file: {full_path}")
                        except OSError: pass
            except Exception as cleanup_err:
                logger.warning(f"Error during URL download cleanup: {cleanup_err}")


def transcribe_audio(audio_path):
    """Transcribes audio using the pre-loaded Faster Whisper model."""
    global WHISPER_MODEL_NAME # Access the globally configured model name

    model_key = WHISPER_MODEL_NAME
    if model_key not in WHISPER_MODELS or WHISPER_MODELS[model_key] is None:
        logger.error(f"Whisper model '{model_key}' not loaded. Cannot transcribe.")
        return None, f"Whisper model '{model_key}' unavailable."

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        logger.error(f"Audio file missing or empty for transcription: {audio_path}")
        return None, "Audio file missing or empty."

    logger.info(f"Starting transcription using '{model_key}' model for: {os.path.basename(audio_path)}")
    start_time = time.time()

    try:
        model = WHISPER_MODELS[model_key]
        # Add vad_filter=True for potentially better accuracy on long silences
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

        detected_language = info.language
        lang_probability = info.language_probability
        logger.info(f"Detected language: {detected_language} (Confidence: {lang_probability:.2f})")
        logger.info(f"Transcription duration: {info.duration:.2f}s")

        # Combine segments into raw transcript - PREPROCESSING HAPPENS LATER
        raw_transcript = " ".join(segment.text.strip() for segment in segments)

        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        logger.debug(f"Raw Transcript generated (first 500 chars): {raw_transcript[:500]}...")

        return raw_transcript, None # Return RAW transcript and no error
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None, f"Transcription failed: {e}"

# --- Text Preprocessing Helpers ---
def remove_noise(text):
    """Removes common filler words."""
    # More conservative list
    text = re.sub(r'\b(um|uh|ah|er|hmm|hmmm|uh huh|um hum)\b\s*', '', text, flags=re.IGNORECASE)
    # Remove extra spaces that might result
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def remove_repeated_words(text):
    """Removes consecutive duplicate words."""
    return re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text, flags=re.IGNORECASE).strip()

def correct_spelling(text):
    """Corrects spelling using pyspellchecker and handles spacing around punctuation."""
    if not SPELL_CHECKER:
        logger.warning("SpellChecker not available, skipping spelling correction.")
        return text

    # Tokenize considering punctuation and contractions
    # This regex attempts to keep contractions together and handle punctuation
    words_and_punct = re.findall(r"[\w'-]+|[.,!?;:]+|\S", text) # \S catches other non-space chars

    corrected_tokens = []
    for token in words_and_punct:
        # Check if it looks like a word (might include hyphen/apostrophe)
        if re.fullmatch(r"[\w'-]+", token) and len(token) > 1: # Avoid correcting single letters typically
            # Check if it's likely an acronym or proper noun (simple check: contains uppercase)
            if not any(c.isupper() for c in token):
                corrected = SPELL_CHECKER.correction(token.lower()) # Correct lowercase version
                # Only use correction if found and different
                if corrected and corrected != token.lower():
                    corrected_tokens.append(corrected)
                else:
                    corrected_tokens.append(token) # Keep original if no good correction
            else:
                corrected_tokens.append(token) # Keep words with uppercase as is
        else:
            corrected_tokens.append(token) # Keep punctuation and other symbols

    # Join tokens, carefully managing spaces
    processed_text = ""
    for i, token in enumerate(corrected_tokens):
        processed_text += token
        # Add space after token unless it's the last one or the next one is punctuation
        if i < len(corrected_tokens) - 1 and \
           not re.fullmatch(r"[.,!?;:]", corrected_tokens[i+1]):
            processed_text += " "

    # Refine spacing around punctuation
    processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text) # Remove space BEFORE specific punctuation
    processed_text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', processed_text) # Add space AFTER punctuation if followed by non-space

    return processed_text.strip()


def preprocess_transcript(transcript):
    """Applies preprocessing steps: lowercase, noise removal, repeat removal, spell check."""
    if not transcript: return ""
    logger.info("Preprocessing transcript...")
    original_length = len(transcript)
    start_time = time.time()

    processed = transcript.lower()
    processed = remove_noise(processed)
    processed = remove_repeated_words(processed)
    processed = correct_spelling(processed) # Includes some punctuation spacing adjustments
    # Capitalize first letter as a final step (spaCy sentence splitting is better handled during chunking)
    processed = processed.capitalize()

    end_time = time.time()
    logger.info(f"Preprocessing finished in {end_time - start_time:.2f}s. Len: {original_length} -> {len(processed)}")
    logger.debug(f"Preprocessed Transcript (first 500 chars): {processed[:500]}...")
    return processed

# --- Chunking Helper (Commented out - Not used in current flow but kept for reference) ---
# def chunk_transcript_with_spacy(transcript):
#     """Splits the transcript into semantic chunks using spaCy."""
#     if not SPACY_NLP:
#         logger.error("spaCy model not loaded. Cannot chunk transcript.")
#         # Basic fallback: split by paragraphs
#         return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
#
#     if not transcript:
#         return []
#
#     logger.info(f"Chunking transcript using spaCy (Chunk Size: {CHUNK_SIZE})...")
#     start_time = time.time()
#     # Process in batches if transcript is very large (e.g., > 1,000,000 chars)
#     # For simplicity, processing all at once here. Ensure server has enough RAM.
#     try:
#         doc = SPACY_NLP(transcript)
#         sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
#     except Exception as e:
#         logger.error(f"spaCy processing failed: {e}. Falling back to paragraph splitting.")
#         return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
#
#
#     chunks = []
#     current_chunk = ""
#     for sentence in sentences:
#         if len(current_chunk) == 0 and len(sentence) <= CHUNK_SIZE:
#              # Start new chunk if empty
#              current_chunk = sentence
#         elif len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
#             # Add sentence to current chunk if it fits
#             current_chunk += " " + sentence
#         elif len(current_chunk) > 0:
#              # Current chunk is full, add it and start new chunk
#              chunks.append(current_chunk.strip())
#              # Handle sentences longer than chunk size
#              if len(sentence) <= CHUNK_SIZE:
#                   current_chunk = sentence
#              else:
#                   logger.warning(f"Sentence length ({len(sentence)}) > CHUNK_SIZE ({CHUNK_SIZE}). Adding as single chunk.")
#                   chunks.append(sentence.strip()) # Add long sentence as its own chunk
#                   current_chunk = "" # Reset
#         else: # current_chunk is empty, but sentence is too long
#              logger.warning(f"Sentence length ({len(sentence)}) > CHUNK_SIZE ({CHUNK_SIZE}). Adding as single chunk.")
#              chunks.append(sentence.strip())
#              current_chunk = ""
#
#     # Add the last chunk if it's not empty
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#
#     end_time = time.time()
#     logger.info(f"Chunking complete in {end_time - start_time:.2f}s. Created {len(chunks)} chunks.")
#     return chunks

# --- RAG Indexing Helper (Commented out - Not used in current flow but kept for reference) ---
# def build_faiss_index(transcript_chunks):
#     """Builds a FAISS index from transcript chunks."""
#     if not EMBEDDING_MODEL or not EMBEDDING_DIMENSION:
#         logger.error("Embedding model/dimension not available. Cannot build FAISS index.")
#         return None, None
#
#     if not transcript_chunks:
#         logger.warning("No transcript chunks provided to build FAISS index.")
#         return None, None
#
#     logger.info(f"Generating embeddings for {len(transcript_chunks)} chunks...")
#     start_time = time.time()
#     try:
#         # Consider batch encoding if many chunks for efficiency
#         embeddings = EMBEDDING_MODEL.encode(transcript_chunks, show_progress_bar=False, batch_size=128) # Adjust batch_size based on GPU VRAM
#         embeddings_np = np.array(embeddings).astype('float32')
#         logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s. Shape: {embeddings_np.shape}")
#
#         logger.info("Building FAISS index (IndexFlatL2)...")
#         index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
#         index.add(embeddings_np)
#         logger.info(f"FAISS index built successfully. Contains {index.ntotal} vectors.")
#         # index object is not directly serializable for cache. Return embeddings.
#         return index, embeddings_np
#     except Exception as e:
#         logger.error(f"Error building FAISS index or generating embeddings: {e}", exc_info=True)
#         return None, None

# --- Ollama Streaming Helper ---
# def load_rag_models(): # Renamed as it's less critical now, but kept for potential reuse
#     global EMBEDDING_MODEL
#     global EMBEDDING_DIMENSION
#     try:
#         from sentence_transformers import SentenceTransformer
#         EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
#         EMBEDDING_DIMENSION = EMBEDDING_MODEL.get_sentence_embedding_dimension()
#         logger.info(f"Embedding models loaded: {settings.EMBEDDING_MODEL_NAME} (dim: {EMBEDDING_DIMENSION})")
#         return True
#     except ImportError:
#         logger.critical("Sentence Transformers library not found. RAG functionality will be disabled.")
#         return False
#     except Exception as e:
#         logger.critical(f"Error loading embedding model {settings.EMBEDDING_MODEL_NAME}: {e}")
#         return False

# # Load Embedding models on startup or when needed (Keep loading, might be needed later)
# if not EMBEDDING_MODEL:
#     load_rag_models()

def get_ollama_response_stream(messages_for_ollama):
    import ollama
    logger.info(f"Initiating Ollama stream with model '{OLLAMA_MODEL}', messages: {messages_for_ollama}")
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama,
            stream=True,
            options={
                'temperature': getattr(settings, 'OLLAMA_TEMPERATURE', 0.7),
                'top_p': getattr(settings, 'OLLAMA_TOP_P', 0.9),
                'num_ctx': getattr(settings, 'OLLAMA_CONTEXT_TOKENS', 4096),
            }
        )
        for part in stream:
            if 'message' in part and 'content' in part['message']: # Added extra check
                yield part['message']['content'].encode('utf-8')
        logger.info("Ollama stream ended.")
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
        yield f"\n\n--- Error: {e} ---".encode('utf-8')
    except Exception as e:
        logger.error(f"Error during Ollama stream: {e}", exc_info=True)
        yield f"\n\n--- Error: {e} ---".encode('utf-8')

def call_llm(messages, model=OLLAMA_MODEL):
    import ollama
    logger.info(f"Calling Ollama model '{model}' with messages: {messages}")
    try:
        response = ollama.chat(model=model, messages=messages)
        logger.info(f"Ollama response received.")
        return response['message']['content']
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}", exc_info=True)
        return f"Error: {e}"

# --- Django Views ---

# View 1: Render Index Page
def index(request):
    """Renders the main chat page (index.html)."""
    # Check critical models - Embedding model is less critical now but kept check for future RAG
    models_ready = all([
        WHISPER_MODELS,
        # SPACY_NLP, # Less critical if not chunking
        # EMBEDDING_MODEL, # Less critical if not doing RAG
        # EMBEDDING_DIMENSION is not None
    ])
    if not models_ready:
         logger.critical("One or more critical models (like Whisper) failed to load. Check logs.")
         # Decide if you want to render an error page or just the index with potential issues
    return render(request, 'index.html')

# View 2: Upload Video (Handles initial question)
@csrf_exempt
def upload_video(request):
    """
    Handles POST video/URL, processes, stores full transcript context in cache,
    clears old history cache, AND optionally asks initial question using the
    full transcript, returning the answer if provided.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    # Essential models check (Whisper is key now)
    if not all([WHISPER_MODELS]): # Removed SPACY_NLP, EMBEDDING_MODEL checks as they are bypassed
        logger.error("Cannot process upload: Essential models (Whisper) not loaded.")
        return JsonResponse({'error': 'Server is not ready. Models not loaded.'}, status=503)

    video_source = None
    source_type = None
    uploaded_file_path = None
    temp_audio_path = None
    session_key = None
    initial_question = None
    processed_transcript = None # Define here for broader scope

    try:
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key
        logger.info(f"Session {session_key[-5:]}: Starting video upload processing...")

        # --- Input Handling (File or URL) ---
        if request.FILES.get('videoFile'):
            source_type = 'file'
            video_file = request.FILES['videoFile']
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
                for chunk in video_file.chunks(): tmp_video.write(chunk)
                uploaded_file_path = tmp_video.name
            video_source = uploaded_file_path
            initial_question = request.POST.get('question', '').strip() # Get question from form data
            logger.info(f"Session {session_key[-5:]}: Processing uploaded file '{video_file.name}'. Initial Q: {'Yes' if initial_question else 'No'}")

        elif request.content_type == 'application/json' and request.body:
             try:
                 data = json.loads(request.body)
                 if data.get('videoUrl'):
                     video_source = data['videoUrl']
                     source_type = 'url'
                     initial_question = data.get('question', '').strip() # Get question from JSON
                     logger.info(f"Session {session_key[-5:]}: Processing URL '{video_source}'. Initial Q: {'Yes' if initial_question else 'No'}")
                 else: return JsonResponse({'error': 'Missing videoUrl in JSON payload'}, status=400)
             except json.JSONDecodeError:
                 logger.warning(f"Session {session_key[-5:]}: Invalid JSON payload received.")
                 return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        else: return JsonResponse({'error': 'No video file or URL JSON payload provided'}, status=400)

        # --- Processing Steps ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_f:
            temp_audio_path = tmp_audio_f.name

        extraction_func = extract_audio_from_file if source_type == 'file' else extract_audio_from_video_url
        if not extraction_func(video_source, temp_audio_path):
            return JsonResponse({'error': 'Audio extraction failed'}, status=500)

        raw_transcript, trans_error = transcribe_audio(temp_audio_path)
        if trans_error: return JsonResponse({'error': trans_error}, status=500)

        processed_transcript = preprocess_transcript(raw_transcript if raw_transcript else "")

        # --- RAG Steps Commented Out ---
        # transcript_chunks = chunk_transcript_with_spacy(processed_transcript)
        # transcript_embeddings_np = np.empty((0, EMBEDDING_DIMENSION), dtype='float32') # Default empty
        # if transcript_chunks:
        #     _, embeddings_temp = build_faiss_index(transcript_chunks)
        #     if embeddings_temp is not None and embeddings_temp.size > 0:
        #          transcript_embeddings_np = embeddings_temp
        #     else:
        #          logger.warning(f"Session {session_key[-5:]}: Chunking succeeded but embedding generation failed or yielded empty results.")
        #          transcript_chunks = [] # Treat as no usable context if embeddings failed
        # else:
        #     logger.warning(f"Session {session_key[-5:]}: No chunks generated (transcript likely empty or preprocessing failed).")
        # --- End RAG Steps Commented Out ---

        # --- Store Full Transcript Context in Cache ---
        # chunks_key = f"transcript_chunks_{session_key}" # Commented out
        # embeddings_key = f"transcript_embeddings_{session_key}" # Commented out
        full_transcript_key = f"full_transcript_{session_key}" # Key for full transcript
        history_key = f"conversation_history_{session_key}"

        # cache.set(chunks_key, transcript_chunks, timeout=CACHE_TIMEOUT) # Commented out
        # cache.set(embeddings_key, transcript_embeddings_np, timeout=CACHE_TIMEOUT) # Commented out
        cache.set(full_transcript_key, processed_transcript, timeout=CACHE_TIMEOUT) # Store full transcript
        cache.delete(history_key) # Clear any old history
        logger.info(f"Session {session_key[-5:]}: Stored full transcript ({len(processed_transcript or '')} chars). Cleared old history cache.") # Log length safely

        # --- Handle Initial Question (If applicable, using FULL transcript) ---
        initial_answer = None
        # Check if we have a processed transcript AND an initial question was asked
        has_video_context = bool(processed_transcript) # Context exists if transcript was processed

        if initial_question and has_video_context:
            logger.info(f"Session {session_key[-5:]}: Processing initial question using full transcript: {initial_question[:50]}...")

            # --- RAG for initial question Commented Out ---
            # relevant_context = ""
            # try:
            #     # Rebuild index from cached embeddings (Now commented out)
            #     # index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            #     # index.add(transcript_embeddings_np)
            #     # question_embedding = EMBEDDING_MODEL.encode([initial_question])[0].astype('float32').reshape(1, -1)
            #     # k = min(RAG_TOP_K, index.ntotal)
            #     # if k > 0:
            #     #     _, indices = index.search(question_embedding, k)
            #     #     if indices.size > 0 and np.all(indices[0] != -1):
            #     #          valid_indices = [i for i in indices[0] if 0 <= i < len(transcript_chunks)]
            #     #          if valid_indices:
            #     #              relevant_context_chunks = [transcript_chunks[i] for i in valid_indices]
            #     #              relevant_context = "\n\n---\n\n".join(relevant_context_chunks)
            #     #              logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} chunks for initial Q.")
            # except Exception as e:
            #     logger.error(f"Session {session_key[-5:]}: RAG error for initial Q: {e}", exc_info=True)
            # --- End RAG for initial question Commented Out ---


            # --- Construct Ollama Payload for Initial Question using FULL Transcript ---
            # Modified prompt to reflect using the full transcript as the primary source
            initial_system_prompt_template = """You are an AI assistant answering questions about the video transcript provided below. Your PRIMARY goal is to answer using ONLY the information within the 'FULL VIDEO TRANSCRIPT'.

1. **Thoroughly search the FULL VIDEO TRANSCRIPT** for the answer to the user's question.
2. If the answer IS found, provide it directly based **strictly** on the transcript text. Do not add outside information.
3. **Only if, after a careful search, you are certain the answer is NOT in the transcript**, state: "This specific detail doesn't seem to be covered in the provided video transcript." Then, ask: "Should I answer using my general knowledge? (Yes/No)". Do NOT provide a general knowledge answer unless the user explicitly agrees.
4. For summary requests ("give summary", "summarize"), provide a summary based *only* on the transcript. For transcript requests ("provide transcript"), reproduce the transcript.
5. Always focus on the user's LATEST question and verify once before answering.

FULL VIDEO TRANSCRIPT:
---
{context_text}
---
                     Answer the user's question below:
                     """
            context_text = processed_transcript if processed_transcript else "(Transcript not available)"
            initial_system_prompt = initial_system_prompt_template.format(context_text=context_text)

            messages = [
                {"role": "system", "content": initial_system_prompt},
                {"role": "user", "content": initial_question}
            ]

            # --- Get FULL response for initial Q (accumulate from stream) ---
            try:
                 full_initial_response = ""
                 # Use the same streaming helper, but collect the full response
                 response_stream = get_ollama_response_stream(messages)
                 for chunk in response_stream:
                     full_initial_response += chunk.decode('utf-8', errors='replace')

                 if "--- Error:" in full_initial_response:
                     error_detail = full_initial_response.split('--- Error:')[1].split('---')[0].strip()
                     initial_answer = f"(AI processing failed for initial question: {error_detail})"
                     logger.error(f"Session {session_key[-5:]}: Ollama failed for initial Q: {error_detail}")
                 elif full_initial_response and "(AI returned no content)" not in full_initial_response:
                     initial_answer = full_initial_response.strip()
                     # NOTE: History is NOT stored server-side anymore. Frontend will handle it.
                     logger.info(f"Session {session_key[-5:]}: Generated initial answer using full transcript.")
                 else:
                     logger.warning(f"Session {session_key[-5:]}: Ollama gave empty/no content for initial Q.")
                     initial_answer = "(AI did not provide an answer to the initial question)"
            except Exception as e:
                 logger.error(f"Session {session_key[-5:]}: Ollama call failed for initial Q: {e}", exc_info=True)
                 initial_answer = "(Error getting initial answer from AI)"

        elif initial_question: # Question asked but no context available
             logger.warning(f"Session {session_key[-5:]}: Cannot answer initial question - transcript processing failed or yielded no content.")
             initial_answer = "(Cannot answer initial question as video processing yielded no usable text content)"

        # --- Final Response ---
        # Send status message and the initial answer (if generated)
        response_data = {'message': 'Video processed successfully. Ready for questions.'}
        if initial_answer:
            response_data['answer'] = initial_answer # Frontend will use this to start history

        return JsonResponse(response_data)

    # --- Error Handling & Cleanup ---
    except Exception as e:
        logger.error(f"Unexpected error in upload_video (Session: {session_key[-5:]}): {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred during processing.'}, status=500)
    finally:
        # Ensure cleanup happens even if errors occur
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except OSError as e: logger.warning(f"Could not remove temp audio {temp_audio_path}: {e}")
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try: os.remove(uploaded_file_path)
            except OSError as e: logger.warning(f"Could not remove temp video {uploaded_file_path}: {e}")
        logger.info(f"Session {session_key[-5:]}: Video upload processing finished.")


@csrf_exempt
def ask_question(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        question = data.get('question', '')
        client_history = data.get('history', [])
        session_key = request.session.session_key
        if not session_key:
            return JsonResponse({'error': 'Session key not found'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if not question.strip():
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    # Embedding model check commented out - RAG bypassed
    # if not EMBEDDING_MODEL or EMBEDDING_DIMENSION is None:
    #     error_message = "RAG models are not loaded. Please ensure the server is configured correctly."
    #     logger.error(f"Session {session_key[-5:]}: {error_message}")
    #     return StreamingHttpResponse(iter([error_message.encode('utf-8')]),
    #                                  content_type="text/plain; charset=utf-8", status=503)

    # --- Retrieve Full Transcript Context from Cache ---
    # chunks_key = f"transcript_chunks_{session_key}" # Commented out
    # embeddings_key = f"transcript_embeddings_{session_key}" # Commented out
    full_transcript_key = f"full_transcript_{session_key}" # Get full transcript key

    # transcript_chunks = cache.get(chunks_key) # Commented out
    # transcript_embeddings_np = cache.get(embeddings_key) # Commented out
    full_transcript = cache.get(full_transcript_key) # Retrieve full transcript

    # --- Check if Video Context (Full Transcript) Exists in Cache ---
    # has_video_context = transcript_chunks is not None and transcript_embeddings_np is not None and transcript_embeddings_np.size > 0 # Old RAG check
    has_full_transcript_context = full_transcript is not None # Check if full transcript exists

    if not has_full_transcript_context:
        logger.warning(f"Session {session_key[-5:]}: No full transcript found in cache. Asking user to upload.")
        no_context_message = "It looks like no video has been processed or the transcript is unavailable. Please upload a video file or provide a video URL first."
        return StreamingHttpResponse(iter([no_context_message.encode('utf-8')]),
                                     content_type="text/plain; charset=utf-8", status=200) # Return 200 as it's informational

    # Modified System Prompt - Always refers to FULL TRANSCRIPT
    system_prompt_template = """You are an AI assistant answering questions about the video transcript provided below. Your PRIMARY goal is to answer using ONLY the information within the 'FULL VIDEO TRANSCRIPT'.

1. **Thoroughly search the FULL VIDEO TRANSCRIPT** for the answer to the user's question.
2. If the answer IS found, provide it directly based **strictly** on the transcript text. Do not add outside information.
3. **Only if, after a careful search, you are certain the answer is NOT in the transcript**, state: "This specific detail doesn't seem to be covered in the provided video transcript." Then, ask: "Should I answer using my general knowledge? (Yes/No)". Do NOT provide a general knowledge answer unless the user explicitly agrees.
4. For summary requests ("give summary", "summarize"), provide a summary based *only* on the transcript. For transcript requests ("provide transcript"), reproduce the transcript.
5. Always focus on the user's LATEST question and verify once before answering.

FULL VIDEO TRANSCRIPT:
---
{context_text}
---
             """

    # Use full transcript as context text
    context_text = full_transcript if full_transcript else "(Full transcript not available)"

    # --- Handle Summarization Request (Simplified, as context is always full transcript) ---
    # The check remains useful to potentially use a different user message for summarization.
    is_summary_request = question.lower() in ["summarize", "summarise", "summarize the video", "summarise the video", "give me a summary", "tl;dr", "tldr"]
    if is_summary_request:
        logger.info(f"Session {session_key[-5:]}: Summarization requested (using full transcript).")
        # User question for summary - can be more specific if needed
        user_message_for_summary = "Provide a comprehensive summary of the video transcript."
        question_to_llm = user_message_for_summary # Override original question if summarizing
    else:
        question_to_llm = question # Use the user's original question

    # --- RAG Search Logic Commented Out ---
    # relevant_context = ""
    # use_full_transcript = False # Flag no longer needed, always true conceptually
    # if question.lower().startswith("analyze full transcript"):
    #     use_full_transcript = True
    #     question = question[len("analyze full transcript"):].strip() # Remove command from question
    #     logger.info(f"Session {session_key[-5:]}: User requested analysis of the full transcript.")
    # elif not has_video_context: # Old check based on RAG context
    #     use_full_transcript = True
    #     logger.warning(f"Session {session_key[-5:]}: No video context for RAG, defaulting to full transcript analysis.")

    # if has_video_context and not use_full_transcript: # Old RAG condition
    #     try:
    #         # if transcript_embeddings_np.shape[1] != EMBEDDING_DIMENSION:
    #         #     logger.error(f"Session {session_key[-5:]}: Embedding dimension mismatch in cache! Expected {EMBEDDING_DIMENSION}, Got {transcript_embeddings_np.shape[1]}. Cannot perform RAG.")
    #         # else:
    #         #     index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    #         #     index.add(transcript_embeddings_np)
    #         #     logger.debug(f"Session {session_key[-5:]}: FAISS index rebuilt with {index.ntotal} vectors for RAG.")
    #         #
    #         #     question_embedding = EMBEDDING_MODEL.encode([question])[0].astype('float32').reshape(1, -1)
    #         #     k = min(RAG_TOP_K, index.ntotal)
    #         #
    #         #     if k > 0:
    #         #         distances, indices = index.search(question_embedding, k)
    #         #         if indices.size > 0 and np.all(indices[0] != -1):
    #         #             valid_indices = [i for i in indices[0] if 0 <= i < len(transcript_chunks)]
    #         #             if valid_indices:
    #         #                 relevant_context_chunks = [transcript_chunks[i] for i in valid_indices]
    #         #                 relevant_context = "\n\n---\n\n".join(relevant_context_chunks)
    #         #                 logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} relevant chunks.")
    #         #             else:
    #         #                 logger.warning(f"Session {session_key[-5:]}: RAG search returned indices out of bounds: {indices[0]}.")
    #         #         else:
    #         #             logger.info(f"Session {session_key[-5:]}: RAG search did not find any relevant chunks (indices: {indices}).")
    #         #     else:
    #         #         logger.info(f"Session {session_key[-5:]}: Not enough vectors in index (or k=0) for RAG search.")
    #     except Exception as e:
    #         logger.error(f"Session {session_key[-5:]}: RAG search failed: {e}", exc_info=True)
            # Continue without relevant context, prompt handles this
    # --- End RAG Search Logic Commented Out ---


    # --- Construct Ollama Payload (Always uses full transcript now) ---
    # context_text = "" # Reset and assign based on full transcript
    # if use_full_transcript and full_transcript: # Condition simplified
    #     context_text = full_transcript
    #     logger.info(f"Session {session_key[-5:]}: Using the full transcript for the query.")
    # elif not use_full_transcript: # Old RAG path
    #     context_text = relevant_context if relevant_context else "(No relevant excerpts found for this question)"
    # elif use_full_transcript and not full_transcript: # Condition simplified
    #     context_text = "(Full transcript not available)"
    #     logger.warning(f"Session {session_key[-5:]}: Requested full transcript analysis but it's not available in cache.")

    final_prompt = system_prompt_template.format(context_text=context_text) # Format with the full transcript

    messages_for_ollama = [{"role": "system", "content": final_prompt}]
    # Limit history turns based on settings
    limited_history = client_history[-(MAX_HISTORY_TURNS * 2):]
    messages_for_ollama.extend(limited_history)
    messages_for_ollama.append({"role": "user", "content": question_to_llm}) # Use appropriate question (original or summary)

    # --- Stream Response ---
    try:
        response_stream = get_ollama_response_stream(messages_for_ollama)
        response = StreamingHttpResponse(response_stream, content_type="text/plain; charset=utf-8")
        # Set headers to prevent caching of the stream
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        logger.info(f"Session {session_key[-5:]}: Streaming response initiated (using full transcript context).")
        return response
    except Exception as e:
        logger.error(f"Session {session_key[-5:]}: Failed to initiate Ollama stream: {e}", exc_info=True)
        error_stream = iter([b"\n\n--- Error: Failed to communicate with AI model. ---"])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=500)