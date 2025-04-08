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
import numpy as np
import faiss # For RAG
import spacy # For sentence splitting
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer # For embeddings
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
EMBEDDING_DIMENSION = None

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

# Download/Load spaCy Model
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

# Load Sentence Transformer Model
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
CHUNK_SIZE = getattr(settings, 'TRANSCRIPT_CHUNK_SIZE', 512)
RAG_TOP_K = getattr(settings, 'RAG_TOP_K', 3)
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
    ffmpeg_location = FFMPEG_COMMAND if FFMPEG_COMMAND != 'ffmpeg' else None

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

# --- Chunking Helper ---
def chunk_transcript_with_spacy(transcript):
    """Splits the transcript into semantic chunks using spaCy."""
    if not SPACY_NLP:
        logger.error("spaCy model not loaded. Cannot chunk transcript.")
        # Basic fallback: split by paragraphs
        return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]

    if not transcript:
        return []

    logger.info(f"Chunking transcript using spaCy (Chunk Size: {CHUNK_SIZE})...")
    start_time = time.time()
    # Process in batches if transcript is very large (e.g., > 1,000,000 chars)
    # For simplicity, processing all at once here. Ensure server has enough RAM.
    try:
        doc = SPACY_NLP(transcript)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        logger.error(f"spaCy processing failed: {e}. Falling back to paragraph splitting.")
        return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]


    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) == 0 and len(sentence) <= CHUNK_SIZE:
             # Start new chunk if empty
             current_chunk = sentence
        elif len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
            # Add sentence to current chunk if it fits
            current_chunk += " " + sentence
        elif len(current_chunk) > 0:
             # Current chunk is full, add it and start new chunk
             chunks.append(current_chunk.strip())
             # Handle sentences longer than chunk size
             if len(sentence) <= CHUNK_SIZE:
                  current_chunk = sentence
             else:
                  logger.warning(f"Sentence length ({len(sentence)}) > CHUNK_SIZE ({CHUNK_SIZE}). Adding as single chunk.")
                  chunks.append(sentence.strip()) # Add long sentence as its own chunk
                  current_chunk = "" # Reset
        else: # current_chunk is empty, but sentence is too long
              logger.warning(f"Sentence length ({len(sentence)}) > CHUNK_SIZE ({CHUNK_SIZE}). Adding as single chunk.")
              chunks.append(sentence.strip())
              current_chunk = ""

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    end_time = time.time()
    logger.info(f"Chunking complete in {end_time - start_time:.2f}s. Created {len(chunks)} chunks.")
    return chunks

# --- RAG Indexing Helper ---
def build_faiss_index(transcript_chunks):
    """Builds a FAISS index from transcript chunks."""
    if not EMBEDDING_MODEL or not EMBEDDING_DIMENSION:
        logger.error("Embedding model/dimension not available. Cannot build FAISS index.")
        return None, None

    if not transcript_chunks:
        logger.warning("No transcript chunks provided to build FAISS index.")
        return None, None

    logger.info(f"Generating embeddings for {len(transcript_chunks)} chunks...")
    start_time = time.time()
    try:
        # Consider batch encoding if many chunks for efficiency
        embeddings = EMBEDDING_MODEL.encode(transcript_chunks, show_progress_bar=False, batch_size=128) # Adjust batch_size based on GPU VRAM
        embeddings_np = np.array(embeddings).astype('float32')
        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s. Shape: {embeddings_np.shape}")

        logger.info("Building FAISS index (IndexFlatL2)...")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(embeddings_np)
        logger.info(f"FAISS index built successfully. Contains {index.ntotal} vectors.")
        # index object is not directly serializable for cache. Return embeddings.
        return index, embeddings_np
    except Exception as e:
        logger.error(f"Error building FAISS index or generating embeddings: {e}", exc_info=True)
        return None, None

# --- Ollama Streaming Helper ---
def get_ollama_response_stream(messages_for_ollama):
    """
    Streams response from Ollama and yields utf-8 encoded content chunks for StreamingHttpResponse.
    """
    logger.info(f"Streaming request to Ollama model '{OLLAMA_MODEL}'. Messages: {len(messages_for_ollama)}")

    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama,
            stream=True
        )

        yielded_something = False

        for chunk in stream:
            message_chunk = chunk.get('message', {})
            content_chunk = message_chunk.get('content', '')

            if content_chunk:
                yielded_something = True
                yield content_chunk.encode('utf-8')  # Important: encode each small chunk

        if not yielded_something:
            logger.warning("Ollama stream finished but yielded no content.")
            yield " (AI returned no content)".encode('utf-8')

        logger.info("Ollama stream finished successfully.")

    except ollama.ResponseError as e:
        error_message = f"Ollama API Error ({e.status_code}): {e.error}"
        logger.error(error_message)
        yield f"\n\n--- Error: {error_message} ---".encode('utf-8')

    except Exception as e:
        logger.error(f"Unexpected error during Ollama streaming: {e}", exc_info=True)
        yield b"\n\n--- Error communicating with AI model. ---"


# --- Django Views ---

# View 1: Render Index Page
def index(request):
    """Renders the main chat page (index.html)."""
    models_ready = all([
        WHISPER_MODELS,
        SPACY_NLP,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION is not None
    ])
    if not models_ready:
         logger.critical("One or more critical models failed to load. Check logs.")
    return render(request, 'index.html')

# View 2: Upload Video (Handles initial question)
@csrf_exempt
def upload_video(request):
    """
    Handles POST video/URL, processes, stores context in cache, clears old history cache,
    AND optionally asks initial question, returning the answer if provided.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    if not all([WHISPER_MODELS, SPACY_NLP, EMBEDDING_MODEL, EMBEDDING_DIMENSION]):
         logger.error("Cannot process upload: Essential models not loaded.")
         return JsonResponse({'error': 'Server is not ready. Models not loaded.'}, status=503)

    video_source = None
    source_type = None
    uploaded_file_path = None
    temp_audio_path = None
    session_key = None
    initial_question = None

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
        transcript_chunks = chunk_transcript_with_spacy(processed_transcript)

        transcript_embeddings_np = np.empty((0, EMBEDDING_DIMENSION), dtype='float32') # Default empty
        if transcript_chunks:
            _, embeddings_temp = build_faiss_index(transcript_chunks)
            if embeddings_temp is not None and embeddings_temp.size > 0:
                 transcript_embeddings_np = embeddings_temp
            else:
                logger.warning(f"Session {session_key[-5:]}: Chunking succeeded but embedding generation failed or yielded empty results.")
                transcript_chunks = [] # Treat as no usable context if embeddings failed
        else:
            logger.warning(f"Session {session_key[-5:]}: No chunks generated (transcript likely empty or preprocessing failed).")


        # --- Store Context in Cache ---
        chunks_key = f"transcript_chunks_{session_key}"
        embeddings_key = f"transcript_embeddings_{session_key}"
        history_key = f"conversation_history_{session_key}" # Key for old cache mechanism (clear it)

        cache.set(chunks_key, transcript_chunks, timeout=CACHE_TIMEOUT)
        cache.set(embeddings_key, transcript_embeddings_np, timeout=CACHE_TIMEOUT)
        # **CRITICAL: Delete the old history cache key. History is now managed client-side.**
        cache.delete(history_key)
        logger.info(f"Session {session_key[-5:]}: Stored {len(transcript_chunks)} chunks and embeddings (Shape: {transcript_embeddings_np.shape}). Cleared any old server-side history cache.")

        # --- Handle Initial Question (If applicable) ---
        initial_answer = None
        # Check if we have context AND an initial question was asked
        has_video_context = transcript_chunks and transcript_embeddings_np.size > 0
        if initial_question and has_video_context:
            logger.info(f"Session {session_key[-5:]}: Processing initial question: {initial_question[:50]}...")
            relevant_context = ""
            # --- RAG for initial question ---
            try:
                index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                index.add(transcript_embeddings_np)
                question_embedding = EMBEDDING_MODEL.encode([initial_question])[0].astype('float32').reshape(1, -1)
                k = min(RAG_TOP_K, index.ntotal)
                if k > 0:
                    _, indices = index.search(question_embedding, k)
                    if indices.size > 0 and np.all(indices[0] != -1):
                         valid_indices = [i for i in indices[0] if 0 <= i < len(transcript_chunks)]
                         if valid_indices:
                             relevant_context_chunks = [transcript_chunks[i] for i in valid_indices]
                             relevant_context = "\n\n---\n\n".join(relevant_context_chunks)
                             logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} chunks for initial Q.")
            except Exception as e:
                 logger.error(f"Session {session_key[-5:]}: RAG error for initial Q: {e}", exc_info=True)

            # --- Construct Ollama Payload for Initial Question ---
            # Use a simpler prompt for the *first* question, similar to the refined ask_question prompt
            # but without needing history.
            initial_system_prompt_template = """You are an AI assistant answering a question about a video transcript.
                Base your answer PRIMARILY on the 'RELEVANT TRANSCRIPT EXCERPTS' provided below.
                IMPORTANT RULES:
                1. If the answer IS found in the excerpts above, provide it directly based on them.
                2. If the answer is NOT found in the excerpts above, state "This topic doesn't seem to be covered in the provided video excerpts." and then ask "Should I answer using my general knowledge? (Yes/No)". Do NOT answer from general knowledge unless the user replies "Yes".
                RELEVANT TRANSCRIPT EXCERPTS:
                ---
                {context_text}
                ---
                Answer the user's question below:
                """
            context_text = relevant_context if relevant_context else "(None found/relevant)"
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
                      logger.info(f"Session {session_key[-5:]}: Generated initial answer.")
                 else:
                      logger.warning(f"Session {session_key[-5:]}: Ollama gave empty/no content for initial Q.")
                      initial_answer = "(AI did not provide an answer to the initial question)"
            except Exception as e:
                 logger.error(f"Session {session_key[-5:]}: Ollama call failed for initial Q: {e}", exc_info=True)
                 initial_answer = "(Error getting initial answer from AI)"

        elif initial_question: # Question asked but no context available
             logger.warning(f"Session {session_key[-5:]}: Cannot answer initial question - transcript context missing or embeddings failed.")
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
    """
    Handles POST question with history, performs RAG, calls Ollama with refined prompt,
    and STREAMS the response. Uses history provided by the client.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    session_key = request.session.session_key
    if not session_key:
        return StreamingHttpResponse(iter([b"--- Error: Session not found or expired. Please upload video again. ---"]),
                                     content_type="text/plain; charset=utf-8", status=400)

    if not all([EMBEDDING_MODEL, EMBEDDING_DIMENSION is not None]):
        logger.error(f"Session {session_key[-5:]}: Cannot process question: RAG models not loaded.")
        return StreamingHttpResponse(iter([b"--- Error: Server is not ready (models not loaded). ---"]),
                                     content_type="text/plain; charset=utf-8", status=503)

    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        # ** Get history from client **
        client_history = data.get('history', []) # Expects list of {"role": "...", "content": "..."}

        if not question:
            return StreamingHttpResponse(iter([b"--- Error: No question provided. ---"]),
                                         content_type="text/plain; charset=utf-8", status=400)
    except json.JSONDecodeError:
         return StreamingHttpResponse(iter([b"--- Error: Invalid JSON in request body. ---"]),
                                      content_type="text/plain; charset=utf-8", status=400)

    logger.info(f"Session {session_key[-5:]}: Processing question (streaming) with {len(client_history)} history entries: {question[:50]}...")

    # --- Retrieve Context (Chunks & Embeddings) from Cache ---
    chunks_key = f"transcript_chunks_{session_key}"
    embeddings_key = f"transcript_embeddings_{session_key}"

    transcript_chunks = cache.get(chunks_key)
    transcript_embeddings_np = cache.get(embeddings_key)

    # --- Check if Video Context Exists in Cache ---
    has_video_context = transcript_chunks is not None and transcript_embeddings_np is not None and transcript_embeddings_np.size > 0

    if not has_video_context:
        logger.warning(f"Session {session_key[-5:]}: No video context found in cache. Asking user to upload.")
        # Stream back a predefined message without calling Ollama
        no_context_message = "It looks like no video has been provided. Please upload a video file or provide a video URL first."
        return StreamingHttpResponse(iter([no_context_message.encode('utf-8')]),
                                     content_type="text/plain; charset=utf-8", status=200) # Send 200 OK, but with instruction

    # --- Handle Summarization Request (Checks client_history) ---
    if question.lower() in ["summarize", "summarise", "summarize the video", "summarise the video", "give me a summary", "tl;dr", "tldr"]:
        logger.info(f"Session {session_key[-5:]}: Summarization requested.")
        # Use the client-provided history for summarization
        summary_stream = summarize_conversation(client_history) # Pass client_history
        response = StreamingHttpResponse(summary_stream, content_type="text/plain; charset=utf-8")
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response

    # --- Perform RAG Search (If context exists) ---
    relevant_context = ""
    if has_video_context: # Double check, though covered above
        try:
            if transcript_embeddings_np.shape[1] != EMBEDDING_DIMENSION:
                 logger.error(f"Session {session_key[-5:]}: Embedding dimension mismatch in cache! Expected {EMBEDDING_DIMENSION}, Got {transcript_embeddings_np.shape[1]}. Cannot perform RAG.")
            else:
                index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                index.add(transcript_embeddings_np)
                logger.debug(f"Session {session_key[-5:]}: FAISS index rebuilt with {index.ntotal} vectors for RAG.")

                question_embedding = EMBEDDING_MODEL.encode([question])[0].astype('float32').reshape(1, -1)
                k = min(RAG_TOP_K, index.ntotal)

                if k > 0:
                    distances, indices = index.search(question_embedding, k)
                    if indices.size > 0 and np.all(indices[0] != -1):
                        valid_indices = [i for i in indices[0] if 0 <= i < len(transcript_chunks)]
                        if valid_indices:
                            relevant_context_chunks = [transcript_chunks[i] for i in valid_indices]
                            relevant_context = "\n\n---\n\n".join(relevant_context_chunks)
                            logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} relevant chunks.")
                        else:
                             logger.warning(f"Session {session_key[-5:]}: RAG search returned indices out of bounds: {indices[0]}.")
                    else:
                        logger.info(f"Session {session_key[-5:]}: RAG search did not find any relevant chunks (indices: {indices}).")
                else:
                     logger.info(f"Session {session_key[-5:]}: Not enough vectors in index (or k=0) for RAG search.")

        except Exception as e:
             logger.error(f"Session {session_key[-5:]}: RAG search failed: {e}", exc_info=True)
             # Continue without relevant context, prompt handles this

    # --- Construct Ollama Payload with Refined Prompt ---
    system_prompt_template = """You are an AI assistant answering questions about a video transcript.
        Focus ONLY on the user's LAST question which follows the conversation history.
        Use the conversation history for context. Base your answer PRIMARILY on the 'RELEVANT TRANSCRIPT EXCERPTS' provided below.
        IMPORTANT RULES:
        1. If the answer IS found in the excerpts above, provide it directly based on them.
        2. If the answer is NOT found in the excerpts above, state "This topic doesn't seem to be covered in the provided video excerpts." AND THEN ask "Should I answer using my general knowledge? (Yes/No)". Do NOT provide the answer from general knowledge in this response. Wait for the user to say "Yes".And this is applied for all questions tha user asks to you.
        3. Do NOT repeat previous questions or answers from the conversation history. Just answer the user's LAST question according to rules 1 & 2.
        RELEVANT TRANSCRIPT EXCERPTS (primary source for the upcoming question):
        ---
        {context_text}
        ---
        """

    context_text = relevant_context if relevant_context else "(None found/relevant for this question)"
    final_system_prompt = system_prompt_template.format(context_text=context_text)

    messages_for_ollama = [{"role": "system", "content": final_system_prompt}]

    # Add historical context (client provided, limited length)
    limited_history = client_history[-(MAX_HISTORY_TURNS * 2):] # Keep last N user/assistant pairs
    messages_for_ollama.extend(limited_history)

    # Add the actual user question LAST
    messages_for_ollama.append({"role": "user", "content": question})

    # --- Stream Response ---
    try:
        response_stream = get_ollama_response_stream(messages_for_ollama)
        response = StreamingHttpResponse(response_stream, content_type="text/plain; charset=utf-8")
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        logger.info(f"Session {session_key[-5:]}: Streaming response initiated.")
        return response
    except Exception as e:
        logger.error(f"Session {session_key[-5:]}: Failed to initiate Ollama stream: {e}", exc_info=True)
        error_stream = iter([b"\n\n--- Error: Failed to communicate with AI model. ---"])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=500)

# --- Helper function: Summarize conversation ---
def summarize_conversation(conversation_history):
    """
    Generates a summary of the conversation history using Ollama (streaming).
    """
    if not conversation_history:
        yield "No conversation history found to summarize.".encode('utf-8')
        return

    logger.info("Generating conversation summary...")
    # Keep summary prompt simple
    summary_prompt = "Summarize the key points discussed in the following conversation history:"

    # Use the standard chat endpoint for summarization too
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled at summarizing conversations concisely."},
        {"role": "user", "content": summary_prompt}
    ]
    # Append the actual history for the LLM to process
    messages.extend(conversation_history)


    # Use the existing streaming function
    yield from get_ollama_response_stream(messages)

# --- Helper function: call_llm ---

def call_llm(messages, model="llama3.2"):
    """
    Sends a prompt to the LLM and returns the response.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.3}
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return None

def get_ollama_response_stream(messages_for_ollama):
    """
    Gets response from Ollama as a stream and yields content chunks (encoded utf-8).
    """
    logger.info(f"Streaming request to Ollama model '{OLLAMA_MODEL}'. Messages: {len(messages_for_ollama)}")
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama,
            stream=True,
            options={
                "temperature": 0.4,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        )
        yielded_something = False
        for chunk in stream:
            message_chunk = chunk.get('message', {})
            content_chunk = message_chunk.get('content', '')
            if content_chunk:
                yielded_something = True
                yield content_chunk.encode('utf-8')  # Encode properly
        if not yielded_something:
            logger.warning("Ollama stream finished without yielding any content.")
            yield " (AI returned no content)".encode('utf-8')

    except ollama.ResponseError as e:
        error_message = f"Ollama API Error ({e.status_code}): {e.error}"
        logger.error(error_message)
        yield f"\n\n--- Error: {error_message} ---".encode('utf-8')
    except Exception as e:
        logger.error(f"Error during Ollama stream communication: {e}", exc_info=True)
        yield f"\n\n--- Error communicating with AI Model. ---".encode('utf-8')
