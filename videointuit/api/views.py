# api/views.py

from django.core.cache import cache
from django.http import StreamingHttpResponse, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import time
import logging
import os
import re
import subprocess
import tempfile
import json
import nltk  # Corrected import
import torch
import yt_dlp
import ollama
import numpy as np
import faiss
import spacy
import requests
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- YouTube API Key Handling ---
YOUTUBE_API_KEY = None # Default to None
try:
    # Attempt to import from config.py first (recommended for local dev)
    from config import YOUTUBE_API_KEY
    logger.info("Successfully imported YOUTUBE_API_KEY from config.py")
except ImportError:
    logger.warning("Could not import YOUTUBE_API_KEY from config.py. Trying environment variable.")
    # Fallback to environment variable (recommended for production)
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", None)
    if YOUTUBE_API_KEY:
        logger.info("Found YOUTUBE_API_KEY in environment variables.")
    else:
        logger.error("CRITICAL: YOUTUBE_API_KEY not found in config.py or environment variables. YouTube search features will be disabled.")
except Exception as e:
     logger.error(f"An unexpected error occurred during YOUTUBE_API_KEY loading: {e}")


# --- Global Model Loading (Load Once at Startup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
WHISPER_MODELS = {}
SPACY_NLP = None
EMBEDDING_MODEL = None
SPELL_CHECKER = None
EMBEDDING_DIMENSION = None
WHISPER_MODEL_NAME = getattr(settings, 'WHISPER_MODEL_NAME', "medium") # Define here for later use

# Load Whisper Model
try:
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' (Device: {DEVICE}, Compute: {COMPUTE_TYPE})...")
    WHISPER_MODELS[WHISPER_MODEL_NAME] = WhisperModel(
        WHISPER_MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Whisper model '{WHISPER_MODEL_NAME}': {e}", exc_info=True)

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
except Exception as e:
    logger.error(f"CRITICAL: Failed to load spaCy model: {e}", exc_info=True)

# Load Sentence Transformer Model
try:
    EMBEDDING_MODEL_NAME = getattr(settings, 'EMBEDDING_MODEL_NAME', "all-MiniLM-L6-v2")
    logger.info(f"Loading Sentence Transformer model '{EMBEDDING_MODEL_NAME}'...")
    EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    EMBEDDING_DIMENSION = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    logger.info(f"Sentence Transformer model '{EMBEDDING_MODEL_NAME}' loaded (Dim: {EMBEDDING_DIMENSION}).")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load Sentence Transformer model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)

# Initialize Spell Checker
try:
    logger.info("Initializing SpellChecker...")
    SPELL_CHECKER = SpellChecker()
    logger.info("Spell Checker initialized.")
except Exception as e:
    logger.error(f"Failed to initialize SpellChecker: {e}", exc_info=True)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    try:
        nltk.download('punkt', quiet=True)
        logger.info("'punkt' downloaded.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt': {e}")
except Exception as e:
    logger.error(f"Error checking NLTK data: {e}")


# --- Configuration ---
FFMPEG_COMMAND = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
CACHE_TIMEOUT = getattr(settings, 'CHAT_CACHE_TIMEOUT', 3600) # 1 hour cache
OLLAMA_MODEL = getattr(settings, 'OLLAMA_MODEL', 'llama3.2') # Your preferred Ollama model
CHUNK_SIZE = getattr(settings, 'TRANSCRIPT_CHUNK_SIZE', 300) # Smaller chunk size for RAG might be better
RAG_TOP_K = getattr(settings, 'RAG_TOP_K', 4) # Number of YouTube chunks to retrieve
MAX_HISTORY_TURNS = getattr(settings, 'MAX_HISTORY_TURNS', 10)
YT_INITIAL_SEARCH_RESULTS = getattr(settings, 'YT_INITIAL_SEARCH_RESULTS', 15) # How many YT videos to check initially
YT_TRANSCRIPT_LANGUAGES = getattr(settings, 'YT_TRANSCRIPT_LANGUAGES', ['en']) # Prioritize English transcripts


# --- Helper Functions ---

# --- Audio Extraction Helpers ---
def extract_audio_from_file(video_source_path, target_audio_path):
    """Extracts audio from a local video file using FFmpeg."""
    logger.info(f"Extracting audio from file '{os.path.basename(video_source_path)}'...")
    command = [
        FFMPEG_COMMAND,
        "-i", video_source_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", # Suppress verbose output, show only errors
        "-y", # Overwrite output file without asking
        target_audio_path
    ]
    try:
        process = subprocess.run(command, capture_output=True, check=True, text=True, encoding='utf-8', errors='replace')
        logger.info("Audio extraction from file successful.")
        return True
    except subprocess.CalledProcessError as e:
        # Log the actual error message from FFmpeg
        logger.error(f"FFmpeg Error (File): {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"FFmpeg command ('{FFMPEG_COMMAND}') not found. Make sure FFmpeg is installed and in your system's PATH or configure FFMPEG_COMMAND in settings.")
        return False
    except Exception as e:
        logger.error(f"Unexpected FFmpeg Error (File): {e}", exc_info=True)
        return False

def extract_audio_from_video_url(video_url, target_audio_path):
    """
    Downloads video/audio using yt-dlp, extracts audio to WAV, and gets title.
    Returns (success_boolean, title_string_or_none)
    """
    logger.info(f"Attempting to extract audio and title from URL: {video_url}")
    base_target = target_audio_path.rsplit('.', 1)[0] # Get path without extension
    # Get FFmpeg location correctly from settings or None if default 'ffmpeg'
    ffmpeg_location_setting = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
    ffmpeg_location = ffmpeg_location_setting if ffmpeg_location_setting != 'ffmpeg' else None

    ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'{base_target}.%(ext)s',
    'noplaylist': True,
    'quiet': False,
    'no_warnings': True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
        # Removed 'ffmpeg_args': ['-ar', '16000', '-ac', '1']
    }],
    'ffmpeg_location': ffmpeg_location,
    'retries': 3,
    'socket_timeout': 60,
    'extract_flat': 'discard_in_playlist',
    'forcejson': False,
    'skip_download': False,
    'writethumbnail': False,
    'getfilename': False,
    'gettitle': True,
}

    downloaded_correctly = False
    video_title = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Running yt-dlp extract_info...")
            # Use extract_info to get metadata *and* trigger download/processing
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = info_dict.get('title', None)
            logger.info(f"Extracted Title (if available): {video_title}")

            # Check if the specific target WAV file was created by the post-processor
            if os.path.exists(target_audio_path) and os.path.getsize(target_audio_path) > 0:
                downloaded_correctly = True
                logger.info(f"Audio download/extraction via yt-dlp successful: {target_audio_path}")
                # No need to rename if target_audio_path exists
            else:
                # If the target wasn't created directly, yt-dlp might have created
                # a file like 'base_target.wav'. Check for it and rename.
                intermediate_wav = f"{base_target}.wav"
                if os.path.exists(intermediate_wav) and os.path.getsize(intermediate_wav) > 0:
                     try:
                         os.rename(intermediate_wav, target_audio_path)
                         logger.info(f"Renamed intermediate file {intermediate_wav} to {target_audio_path}")
                         downloaded_correctly = True
                     except OSError as rename_err:
                         logger.error(f"Failed to rename {intermediate_wav} to {target_audio_path}: {rename_err}")
                else:
                     logger.error(f"yt-dlp finished, but neither target WAV ({target_audio_path}) nor intermediate ({intermediate_wav}) found or are empty.")

            # Return status and title regardless of cleanup below
            return downloaded_correctly, video_title

    except yt_dlp.utils.DownloadError as e:
        # Provide more context in error logging
        if "Unsupported URL" in str(e) or "Unable to extract" in str(e):
            logger.warning(f"yt-dlp could not process URL (likely not video/audio): {video_url}. Error: {e}")
        else:
            logger.error(f"yt-dlp download/processing error for {video_url}: {e}")
        return False, None # Indicate failure, no title obtained reliably here

    except Exception as e:
        logger.error(f"Unexpected yt-dlp Error for {video_url}: {e}", exc_info=True)
        return False, None # Indicate failure, no title

    finally:
        # Cleanup Logic: Remove any files starting with base_target EXCEPT the final target_audio_path
        if not downloaded_correctly: # Only cleanup if the desired file wasn't successfully created/renamed
            try:
                parent_dir = os.path.dirname(base_target) or '.'
                base_filename = os.path.basename(base_target)
                target_basename = os.path.basename(target_audio_path) # Get the actual target filename

                for f in os.listdir(parent_dir):
                    # Check if filename starts with the base and is NOT the final target file
                    if f.startswith(base_filename) and f != target_basename:
                        try:
                            full_path = os.path.join(parent_dir, f)
                            if os.path.isfile(full_path): # Make sure it's a file
                                os.remove(full_path)
                                logger.warning(f"Cleaned up intermediate/failed yt-dlp file: {full_path}")
                        except OSError as rm_err:
                             logger.warning(f"Error removing intermediate file {full_path}: {rm_err}")
            except Exception as cleanup_err:
                logger.warning(f"Error during URL download cleanup: {cleanup_err}")


# --- Transcription Helper ---
def transcribe_audio(audio_path):
    """Transcribes audio using the pre-loaded Faster Whisper model."""
    global WHISPER_MODEL_NAME # Ensure it's using the globally defined name
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
        # Use vad_filter for potentially better accuracy on long silences
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

        detected_language = info.language
        lang_probability = info.language_probability
        logger.info(f"Detected language: {detected_language} (Confidence: {lang_probability:.2f})")
        logger.info(f"Transcription duration (reported by model): {info.duration:.2f}s")

        # Combine segments into raw transcript - PREPROCESSING HAPPENS LATER
        # Ensure generator is consumed and texts are stripped properly
        raw_transcript = " ".join(segment.text.strip() for segment in segments if segment.text)

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
    # Using \b for word boundaries, handling potential hyphens in fillers
    text = re.sub(r'\b(um|uh|ah|er|hmm|uh-huh|um-hum)\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with single space
    return text.strip()

def remove_repeated_words(text):
    """Removes consecutive duplicate words."""
    # Added flags=re.IGNORECASE
    return re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text, flags=re.IGNORECASE).strip()

def correct_spelling(text):
    """Corrects spelling using pyspellchecker and handles spacing around punctuation."""
    if not SPELL_CHECKER:
        logger.warning("SpellChecker not available, skipping spelling correction.")
        return text

    # Improved regex to handle contractions and possessives better
    words_and_punct = re.findall(r"[\w'-]+|[.,!?;:]+|\S", text) # Keep contractions, handle punctuation
    corrected_tokens = []

    for token in words_and_punct:
        # Check if it looks like a word (contains letters)
        if re.search(r'[a-zA-Z]', token):
             # Check if it's likely an acronym (all caps, >1 letter) or has internal caps (proper noun)
            if len(token) > 1 and (token.isupper() or any('a' <= c <= 'z' for c in token[1:])):
                 corrected_tokens.append(token) # Keep potential acronyms/proper nouns as is
            else:
                # Correct only if it's likely a misspelling
                corrected = SPELL_CHECKER.correction(token.lower())
                # Add corrected word only if it's different and not None
                corrected_tokens.append(corrected if corrected and corrected != token.lower() else token)
        else:
            # Keep punctuation, numbers, symbols as is
            corrected_tokens.append(token)

    # Join tokens, carefully managing spaces
    processed_text = ""
    for i, token in enumerate(corrected_tokens):
        processed_text += token
        # Add space after token unless:
        # 1. It's the last token
        # 2. The next token is punctuation that shouldn't have a leading space
        # 3. The current token is an opening bracket/quote
        if i < len(corrected_tokens) - 1:
            next_token = corrected_tokens[i+1]
            if not re.fullmatch(r"[.,!?;:)]}\]'", next_token) and \
               not re.fullmatch(r"[([{'\"]", token):
                 # Also handle spaces after hyphens within words carefully if needed, though regex handles basic words
                 processed_text += " "

    # Refine spacing around punctuation (simplified)
    processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text) # Remove space BEFORE common punctuation
    processed_text = re.sub(r'([([{])\s+', r'\1', processed_text) # Remove space AFTER opening brackets
    processed_text = re.sub(r'\s+([)]}])', r'\1', processed_text) # Remove space BEFORE closing brackets

    return processed_text.strip()


def preprocess_transcript(transcript):
    """Applies preprocessing steps: lowercase, noise removal, repeat removal, spell check."""
    if not transcript: return ""
    logger.info("Preprocessing transcript...")
    original_length = len(transcript)
    start_time = time.time()

    processed = transcript.lower() # Start with lowercase
    processed = remove_noise(processed)
    processed = remove_repeated_words(processed)
    # Spell correction can sometimes mess up formatting, apply carefully
    processed = correct_spelling(processed)

    # Capitalize first letter of the whole text
    if processed:
        processed = processed[0].upper() + processed[1:]

    # Optional: Capitalize after sentence-ending punctuation (basic)
    # This regex tries to capitalize the first letter after ., !, ? followed by space(s)
    processed = re.sub(r'([.!?])\s+(\w)', lambda m: m.group(1) + ' ' + m.group(2).upper(), processed)

    end_time = time.time()
    logger.info(f"Preprocessing finished in {end_time - start_time:.2f}s. Len: {original_length} -> {len(processed)}")
    logger.debug(f"Preprocessed Transcript (first 500 chars): {processed[:500]}...")
    return processed


# --- Chunking Helper ---
def chunk_transcript_with_spacy(transcript, chunk_size=CHUNK_SIZE):
    """Splits the transcript into semantic chunks using spaCy sentences."""
    if not SPACY_NLP:
        logger.error("spaCy model not loaded. Cannot chunk transcript using spaCy.")
        # Basic fallback: split by double newline (paragraphs) or fixed length
        fallback_chunks = [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
        if fallback_chunks:
             logger.warning("Falling back to paragraph splitting for chunking.")
             return fallback_chunks
        else:
             # If no paragraphs, split by approx length as last resort
             logger.warning(f"Falling back to fixed-size ({chunk_size}) character splitting for chunking.")
             return [transcript[i:i+chunk_size].strip() for i in range(0, len(transcript), chunk_size) if transcript[i:i+chunk_size].strip()]

    if not transcript:
        return []

    logger.info(f"Chunking transcript using spaCy (Target Chunk Size: ~{chunk_size} chars)...")
    start_time = time.time()

    try:
        # Increase max_length if needed, handle potential memory issues for very large transcripts
        # Consider processing in batches if memory becomes an issue
        # SPACY_NLP.max_length = max(len(transcript) + 100, SPACY_NLP.max_length)
        doc = SPACY_NLP(transcript)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ValueError as e:
        if "max_length" in str(e):
            logger.error(f"spaCy max_length ({SPACY_NLP.max_length}) potentially exceeded. Increase nlp.max_length if needed and RAM allows. Falling back to paragraph splitting. Error: {e}")
        else:
             logger.error(f"spaCy processing error: {e}. Falling back to paragraph splitting.")
        # Fallback same as above
        fallback_chunks = [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
        return fallback_chunks if fallback_chunks else [transcript[i:i+chunk_size].strip() for i in range(0, len(transcript), chunk_size) if transcript[i:i+chunk_size].strip()]
    except Exception as e:
        logger.error(f"Unexpected error during spaCy processing: {e}. Falling back to paragraph splitting.", exc_info=True)
        # Fallback same as above
        fallback_chunks = [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
        return fallback_chunks if fallback_chunks else [transcript[i:i+chunk_size].strip() for i in range(0, len(transcript), chunk_size) if transcript[i:i+chunk_size].strip()]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence_len = len(sentence)
        current_chunk_len = len(current_chunk)

        # If a single sentence is way too long, add it as its own chunk
        # Also check if adding the sentence would grossly exceed the chunk size
        if sentence_len > chunk_size * 1.5: # Allow some flexibility
            if current_chunk: # Add the previous chunk first
                chunks.append(current_chunk)
            logger.warning(f"Sentence length ({sentence_len}) significantly > CHUNK_SIZE ({chunk_size}). Adding as single chunk.")
            chunks.append(sentence)
            current_chunk = "" # Reset
        elif current_chunk_len == 0:
            # Start a new chunk if current is empty
            current_chunk = sentence
        # Check if adding the sentence (plus a space) fits within the chunk size
        elif current_chunk_len + sentence_len + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            # Current chunk is full, finalize it and start new one
            chunks.append(current_chunk)
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    end_time = time.time()
    logger.info(f"Chunking complete in {end_time - start_time:.2f}s. Created {len(chunks)} chunks.")
    return chunks


# --- RAG Indexing Helper ---
def build_faiss_index(transcript_chunks):
    """Builds a FAISS index from transcript chunks."""
    if not EMBEDDING_MODEL or not EMBEDDING_DIMENSION:
        logger.error("Embedding model/dimension not available. Cannot build FAISS index.")
        return None, None # Return None for both index and embeddings_np

    if not transcript_chunks:
        logger.warning("No transcript chunks provided to build FAISS index.")
        return None, None

    logger.info(f"Generating embeddings for {len(transcript_chunks)} chunks...")
    start_time = time.time()

    try:
        # Encode chunks into embeddings
        # Adjust batch_size based on VRAM and typical chunk count
        embeddings = EMBEDDING_MODEL.encode(transcript_chunks, show_progress_bar=False, batch_size=128)
        embeddings_np = np.array(embeddings).astype('float32')

        # Normalize embeddings for cosine similarity if using IndexFlatIP (Inner Product)
        # faiss.normalize_L2(embeddings_np) # Uncomment if using IndexFlatIP

        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s. Shape: {embeddings_np.shape}")

        # Build FAISS index
        logger.info("Building FAISS index (IndexFlatL2)...")
        # Using IndexFlatL2 (Euclidean distance). For cosine similarity, use IndexFlatIP and normalize embeddings.
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        # Add the embeddings to the index
        index.add(embeddings_np)

        logger.info(f"FAISS index built successfully. Contains {index.ntotal} vectors.")
        # Return the built index AND the numpy embeddings (needed for RAG search/cache)
        return index, embeddings_np

    except Exception as e:
        logger.error(f"Error building FAISS index or generating embeddings: {e}", exc_info=True)
        return None, None


# --- Ollama Streaming Helper ---
def get_ollama_response_stream(messages_for_ollama):
    """Yields response chunks from Ollama stream."""
    logger.info(f"Initiating Ollama stream with model '{OLLAMA_MODEL}', messages count: {len(messages_for_ollama)}")
    if messages_for_ollama:
        # Log system prompt carefully (might be large)
        logger.debug(f"Ollama System Prompt (first 300 chars): {messages_for_ollama[0]['content'][:300]}...")
        logger.debug(f"Ollama User Message (last): {messages_for_ollama[-1]['content']}")

    try:
        # Get settings for Ollama call, providing defaults
        ollama_options = {
            'temperature': getattr(settings, 'OLLAMA_TEMPERATURE', 0.7),
            'top_p': getattr(settings, 'OLLAMA_TOP_P', 0.9),
            'num_ctx': getattr(settings, 'OLLAMA_CONTEXT_TOKENS', 4096), # Context window size
            # Add other options as needed, e.g., 'stop': ['\nUser:']
        }

        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama,
            stream=True,
            options=ollama_options
        )

        for part in stream:
            # Check if the message content exists in the response part
            if 'message' in part and 'content' in part['message']:
                # Yield the content chunk, properly encoded
                yield part['message']['content'].encode('utf-8')

            # Check if the stream is done
            if 'done' in part and part['done']:
                logger.info(f"Ollama stream finished. Reason: {part.get('done_reason', 'N/A')}")
                # Log statistics if available and needed
                if 'total_duration' in part:
                    logger.info(f"Ollama total duration: {part['total_duration']/1e9:.2f}s")
                if 'eval_count' in part and 'eval_duration' in part and part['eval_duration'] > 0:
                     logger.info(f"Ollama eval count: {part['eval_count']}, eval rate: {part['eval_count']/(part['eval_duration']/1e9):.2f} t/s")
                break # Exit loop once done

    except ollama.ResponseError as e:
        # Handle specific Ollama API errors
        logger.error(f"Ollama API error: Status {e.status_code}, Error: {e.error}")
        # Yield a user-friendly error message
        yield f"\n\n--- Error: Ollama API Error ({e.status_code}: {e.error}) ---".encode('utf-8')
    except requests.exceptions.ConnectionError as e:
         logger.error(f"Ollama connection error: Could not connect to Ollama instance. Ensure Ollama is running. Error: {e}")
         yield f"\n\n--- Error: Could not connect to AI model service. Please ensure it is running. ---".encode('utf-8')
    except Exception as e:
        # Handle other potential errors during streaming
        logger.error(f"Error during Ollama stream: {e}", exc_info=True)
        yield f"\n\n--- Error: An unexpected issue occurred while communicating with the AI model ---".encode('utf-8')


# --- Ollama Non-Streaming Helper (for keywords) ---
def call_llm(messages, model=OLLAMA_MODEL):
    """Calls Ollama non-streaming and returns the full response content."""
    logger.info(f"Calling Ollama model '{model}' (non-streaming) with {len(messages)} messages.")
    try:
        # Add options if needed, similar to streaming call
        ollama_options = {
            'temperature': getattr(settings, 'OLLAMA_TEMPERATURE', 0.5), # Maybe lower temp for keyword extraction
             'num_ctx': getattr(settings, 'OLLAMA_CONTEXT_TOKENS', 4096),
        }
        response = ollama.chat(model=model, messages=messages, options=ollama_options) # stream=False is default
        logger.info(f"Ollama non-streaming response received.")
        # Check structure carefully
        if 'message' in response and 'content' in response['message']:
             return response['message']['content']
        else:
             logger.error(f"Ollama non-streaming response structure unexpected: {response}")
             return "Error: Received unexpected response structure from AI model"

    except ollama.ResponseError as e:
        logger.error(f"Ollama API error (non-streaming): Status {e.status_code}, Error: {e.error}")
        return f"Error: Ollama API Error ({e.status_code})"
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Ollama connection error (non-streaming): {e}")
        return f"Error: Could not connect to AI model service."
    except Exception as e:
        logger.error(f"Error calling Ollama (non-streaming): {e}", exc_info=True)
        return f"Error: Could not communicate with AI model"


# --- Keyword Extraction Helper ---
def extract_keywords_with_llm(transcript, num_keywords=5):
    """Extracts keywords from transcript using Ollama."""
    if not transcript:
        return ""

    # Limit transcript length sent to LLM for keyword extraction to avoid large payloads
    max_chars_for_keywords = getattr(settings, 'KEYWORD_EXTRACTION_MAX_CHARS', 5000)
    truncated_transcript = transcript[:max_chars_for_keywords]
    if len(transcript) > max_chars_for_keywords:
         logger.warning(f"Transcript truncated to {max_chars_for_keywords} chars for keyword extraction.")


    logger.info(f"Extracting keywords from transcript using LLM...")
    prompt = f"""Analyze the following video transcript and extract the {num_keywords} most important and central keywords or keyphrases. Focus on the specific topics, entities, processes, or concepts that are the *main subject* of the discussion. Identify terms essential for understanding the core content. Avoid broad terms or concepts mentioned only in passing. Do not include keywords unrelated to the main theme, even if important in general.

    List the keywords separated ONLY by commas (e.g., keyword1, keyphrase two, keyword3). Do not add any introduction, explanation, or numbering.

    Transcript:
    ---
    {truncated_transcript}
    ---
    Keywords:"""

    messages = [
        # System prompt could be added for role-playing if needed, but simple task here.
        {"role": "user", "content": prompt}
    ]

    keywords_string = call_llm(messages) # Use the non-streaming call

    if not keywords_string or keywords_string.startswith("Error:"):
        logger.error(f"Keyword extraction failed or returned error: {keywords_string}")
        return "" # Return empty string on failure

    # Basic cleanup: remove potential "Keywords:", leading/trailing whitespace, handle empty strings
    keywords_string = keywords_string.replace("Keywords:", "").strip()
    keywords_list = [kw.strip() for kw in keywords_string.split(',') if kw.strip()] # Split and clean

    logger.info(f"Keywords extracted: {keywords_list}")
    # Return the top N keywords as a comma-separated string
    return ", ".join(keywords_list[:num_keywords])


#--- YouTube Integration Helpers ---
def search_youtube_videos(query, initial_max_results=YT_INITIAL_SEARCH_RESULTS):
    """Searches YouTube for videos based on query."""
    if not YOUTUBE_API_KEY:
        logger.error("Cannot search YouTube: API Key is missing.")
        return []
    if not query:
        logger.warning("Cannot search YouTube: Query is empty.")
        return []

    logger.info(f"Searching YouTube for MEDIUM length videos related to: '{query}'...")
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'videoDuration': 'medium', # ~4-20 minutes
        'relevanceLanguage': YT_TRANSCRIPT_LANGUAGES[0] if YT_TRANSCRIPT_LANGUAGES else 'en', # Prioritize primary language
        'maxResults': min(initial_max_results, 50), # Ensure maxResults is within allowed range (1-50)
        'key': YOUTUBE_API_KEY
    }

    try:
        response = requests.get(search_url, params=params, timeout=15) # 15 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error(f"YouTube search timed out for query: {query}")
        return []
    except requests.exceptions.RequestException as e:
        # This catches connection errors, HTTP errors, etc.
        logger.error(f"Network or HTTP error during YouTube search: {e}")
        # Log response text if available for debugging API key issues etc.
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"YouTube search response text: {e.response.text}")
        return []
    except json.JSONDecodeError:
        logger.error("Failed to decode YouTube search response JSON.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during YouTube search: {e}", exc_info=True)
        return []

    # Check for API errors returned in the JSON payload
    if 'error' in data:
        logger.error(f"YouTube API Error (Search): {data['error'].get('message', 'Unknown Error')}")
        return []

    video_ids = []
    for item in data.get('items', []):
        # Ensure the item is a video and has an ID
        if item.get('id', {}).get('kind') == 'youtube#video' and item.get('id', {}).get('videoId'):
            video_ids.append(item['id']['videoId'])

    logger.info(f"Initial YouTube search found {len(video_ids)} potential video IDs.")
    return video_ids


def get_youtube_video_details(video_ids):
    """Fetches details (title, stats) for a list of YouTube video IDs."""
    if not YOUTUBE_API_KEY:
        logger.error("Cannot get YouTube details: API Key is missing.")
        return {}
    if not video_ids:
        return {}

    logger.info(f"Fetching details for {len(video_ids)} YouTube videos...")
    details_url = 'https://www.googleapis.com/youtube/v3/videos'
    video_details = {}

    # Process in batches of 50 (YouTube API limit)
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        ids_string = ','.join(batch_ids)
        params = {
            'part': 'snippet,statistics', # Get title, description, AND view/like counts
            'id': ids_string,
            'key': YOUTUBE_API_KEY
        }

        try:
            response = requests.get(details_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"YouTube details fetch timed out for batch starting with: {batch_ids[0]}")
            continue # Skip this batch
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error fetching YouTube video details: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 logger.error(f"YouTube details response text: {e.response.text}")
            continue # Skip this batch
        except json.JSONDecodeError:
            logger.error("Failed to decode YouTube details response JSON.")
            continue # Skip this batch
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching YouTube video details: {e}", exc_info=True)
            continue # Skip this batch

        # Check for API errors in the response payload
        if 'error' in data:
            logger.error(f"YouTube API Error (Details): {data['error'].get('message', 'Unknown Error')}")
            # Don't stop processing other batches if one fails
            continue

        # Process items in the current batch
        for item in data.get('items', []):
            video_id = item.get('id')
            if not video_id: continue # Skip if item has no ID

            title = item.get('snippet', {}).get('title', 'N/A')
            # Use .get() with default value 0 in case stats are missing/private
            view_count = int(item.get('statistics', {}).get('viewCount', 0))
            # Like count can be hidden, default to -1 if missing/hidden
            like_count = int(item.get('statistics', {}).get('likeCount', -1))
            # Construct the standard YouTube URL
            url = f"https://www.youtube.com/watch?v={video_id}"

            video_details[video_id] = {
                'id': video_id,
                'title': title,
                'url': url,
                'view_count': view_count,
                'like_count': like_count # -1 indicates hidden or unavailable
            }

    logger.info(f"Successfully fetched details for {len(video_details)} YouTube videos.")
    return video_details


def fetch_youtube_transcript(video_id):
    """Fetches the transcript for a single YouTube video ID using preferred languages."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching in the preferred languages order
        transcript = transcript_list.find_generated_transcript(YT_TRANSCRIPT_LANGUAGES)
        # If not found, uncomment below to try manually created ones as a fallback
        # if not transcript:
        #     transcript = transcript_list.find_manually_created_transcript(YT_TRANSCRIPT_LANGUAGES)

        logger.debug(f"Transcript found for {video_id} (Language: {transcript.language}). Fetching content...")
        full_transcript_list = transcript.fetch()
        # Join the 'text' parts of the fetched transcript segments
        full_transcript = ' '.join([t['text'].strip() for t in full_transcript_list if 'text' in t])
        logger.debug(f"Transcript fetched successfully for {video_id}.")
        return full_transcript

    except (TranscriptsDisabled, NoTranscriptFound):
        logger.debug(f"Transcript not available or disabled for YouTube video {video_id}.")
        return None
    except Exception as e:
        # Log other potential errors like network issues during fetch
        logger.warning(f"Error fetching/processing YouTube transcript for {video_id}: {str(e)}")
        return None

def process_related_youtube_videos(query):
    """Orchestrates the YouTube search, detail fetching, transcript retrieval, preprocessing, and sorting."""
    if not YOUTUBE_API_KEY:
        logger.error("Cannot process related YouTube videos: API Key is missing.")
        return [] # Return empty list

    initial_video_ids = search_youtube_videos(query, initial_max_results=YT_INITIAL_SEARCH_RESULTS)
    if not initial_video_ids:
        logger.info("No potential YouTube videos found in initial search.")
        return []

    video_details = get_youtube_video_details(initial_video_ids)
    if not video_details:
        logger.warning("Could not retrieve details for the found YouTube video IDs.")
        return [] # Cannot proceed without details

    logger.info("Fetching and preprocessing transcripts for potential YouTube videos...")
    valid_videos_with_transcripts = []
    checked_count = 0
    total_to_check = len(video_details)

    for video_id, details in video_details.items():
        checked_count += 1
        logger.debug(f"Checking YouTube video {checked_count}/{total_to_check} (ID: {video_id})...")
        raw_transcript = fetch_youtube_transcript(video_id)

        if raw_transcript:
            # Preprocess the YouTube transcript too
            processed_yt_transcript = preprocess_transcript(raw_transcript)
            if processed_yt_transcript: # Ensure preprocessing didn't result in empty string
                details['transcript'] = processed_yt_transcript # Store the processed transcript
                valid_videos_with_transcripts.append(details)
                logger.debug(f"Usable transcript found and processed for {video_id}.")
            else:
                logger.debug(f"YouTube transcript for {video_id} became empty after preprocessing.")
        # else: transcript fetch failed or returned None (already logged in fetch function)

        # Optional: Add a small delay to avoid hitting API limits too quickly if necessary
        # time.sleep(0.1)

    logger.info(f"Found {len(valid_videos_with_transcripts)} YouTube videos with available & non-empty processed transcripts.")

    if not valid_videos_with_transcripts:
        logger.info("No relevant YouTube videos found with usable transcripts after processing.")
        return []

    # Sort by View Count (descending) as a proxy for relevance/popularity
    sorted_videos = sorted(valid_videos_with_transcripts, key=lambda x: x.get('view_count', 0), reverse=True)

    # Select Top N (default RAG_TOP_K=4)
    final_selection = sorted_videos[:RAG_TOP_K]
    logger.info(f"Selected top {len(final_selection)} YouTube videos based on view count for RAG context.")

    # Return the list of selected video dictionaries
    # (id, title, url, transcript, view_count, like_count)
    return final_selection


# --- Django Views ---

# View 1: Render Index Page
def index(request):
    """Renders the main chat page (index.html)."""
    # Basic check if critical models needed for core functionality are loaded
    models_ready = all([
        WHISPER_MODELS,
        # SPACY_NLP, # Less critical for just rendering the page
        EMBEDDING_MODEL, # Critical for RAG features
        EMBEDDING_DIMENSION is not None
    ])
    context = {'models_ready': models_ready}
    if not models_ready:
        logger.critical("One or more critical models (Whisper/Embedding) failed to load. Chat features requiring them may be limited or unavailable.")
        context['error_message'] = "Server components are initializing or failed to load. Some features might be unavailable."

    return render(request, 'index.html', context)


# View 2: Upload Video
@csrf_exempt
def upload_video(request):
    """
    Handles POST video/URL, processes audio, gets transcript, optionally finds related
    YouTube videos, sets up RAG context in cache, clears old history,
    and optionally answers an initial question.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    # Essential models check for processing
    if not all([WHISPER_MODELS, EMBEDDING_MODEL]): # spaCy less critical if fallback works
        logger.error("Cannot process upload: Essential models (Whisper/Embedding) not loaded.")
        return JsonResponse({'error': 'Server is not ready. Core processing components missing.'}, status=503) # Service Unavailable

    video_source = None
    source_type = None # 'file' or 'url'
    uploaded_file_path = None # Path to temp file if uploaded
    temp_audio_path = None # Path to extracted temp audio
    session_key = None
    initial_question = None
    processed_transcript = None
    youtube_results_for_frontend = [] # Store {title, url} for frontend display
    find_other_videos_flag = False # Default

    try:
        # Ensure session exists
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key
        logger.info(f"Session {session_key[-5:]}: Starting video upload processing...")

        # --- Input Handling (File or URL + Flags) ---
        if request.FILES.get('videoFile'):
            source_type = 'file'
            video_file = request.FILES['videoFile']
            # Create a temporary file to store the upload
            # Keep the original extension for potential format compatibility
            suffix = os.path.splitext(video_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
                for chunk in video_file.chunks():
                    tmp_video.write(chunk)
                uploaded_file_path = tmp_video.name
            video_source = uploaded_file_path
            # Get other form data (initial question, find videos flag)
            initial_question = request.POST.get('question', '').strip()
            find_other_videos_flag = request.POST.get('find_other_videos', 'false').lower() == 'true'
            logger.info(f"Session {session_key[-5:]}: Processing uploaded file '{video_file.name}'. Initial Q: {'Yes' if initial_question else 'No'}. Find YT: {find_other_videos_flag}")

        elif request.content_type == 'application/json' and request.body:
             # Handle URL submission via JSON
             try:
                 data = json.loads(request.body)
                 if data.get('videoUrl'):
                     video_source = data['videoUrl']
                     source_type = 'url'
                     initial_question = data.get('question', '').strip()
                     find_other_videos_flag = data.get('find_other_videos', False) # Get flag from JSON
                     logger.info(f"Session {session_key[-5:]}: Processing URL '{video_source}'. Initial Q: {'Yes' if initial_question else 'No'}. Find YT: {find_other_videos_flag}")
                 else:
                     return JsonResponse({'error': 'Missing videoUrl in JSON payload'}, status=400)
             except json.JSONDecodeError:
                 logger.warning(f"Session {session_key[-5:]}: Invalid JSON payload received.")
                 return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        else:
            # Neither file nor valid JSON URL provided
            return JsonResponse({'error': 'No video file or URL JSON payload provided'}, status=400)


        # --- Processing Steps ---

        # 1. Extract Audio
        # Create a temporary file specifically for the WAV audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_f:
            temp_audio_path = tmp_audio_f.name

        extraction_success = False
        video_title = None # Initialize title
        if source_type == 'file':
            extraction_success = extract_audio_from_file(video_source, temp_audio_path)
        else: # source_type == 'url'
            extraction_success, video_title = extract_audio_from_video_url(video_source, temp_audio_path)

        if not extraction_success:
             # Error logged within extraction functions
            return JsonResponse({'error': 'Audio extraction failed. Check logs for details.'}, status=500)

        # 2. Transcribe Audio
        raw_transcript, trans_error = transcribe_audio(temp_audio_path)
        if trans_error:
            return JsonResponse({'error': trans_error}, status=500)
        if not raw_transcript:
            logger.error(f"Session {session_key[-5:]}: Transcription resulted in empty text.")
            return JsonResponse({'error': 'Transcription resulted in empty text.'}, status=500)

        # 3. Preprocess Transcript
        processed_transcript = preprocess_transcript(raw_transcript)
        if not processed_transcript:
            # Log warning but don't necessarily fail; maybe user wants to chat without context?
            logger.warning(f"Session {session_key[-5:]}: Preprocessing resulted in empty transcript. Proceeding without text context.")
            # Fallback: Use raw transcript if preprocessing empties it?
            # processed_transcript = raw_transcript # Uncomment if you want to use raw if preprocessed is empty

        # 4. Store Base Transcript & Clear History/Old RAG Data
        full_transcript_key = f"full_transcript_{session_key}"
        history_key = f"conversation_history_{session_key}" # Still useful for chat history
        youtube_chunks_key = f"youtube_chunks_{session_key}" # Cache key for YT chunks
        youtube_embeddings_key = f"youtube_embeddings_{session_key}" # Cache key for YT embeddings

        # Store the main transcript (even if empty, to indicate processing happened)
        cache.set(full_transcript_key, processed_transcript, timeout=CACHE_TIMEOUT)
        # Clear any previous history and RAG data for this session
        cache.delete(history_key) # Clear chat history
        cache.delete(youtube_chunks_key)
        cache.delete(youtube_embeddings_key)
        # Clear pending permission state from previous chats
        request.session.pop('pending_general_knowledge_permission', None)
        request.session.pop('original_question_for_general_knowledge', None)
        request.session.save()

        logger.info(f"Session {session_key[-5:]}: Stored base transcript ({len(processed_transcript or '')} chars). Cleared old cache/state.")


        # 5. --- Optional: YouTube Search & RAG Setup ---
        if find_other_videos_flag and YOUTUBE_API_KEY and processed_transcript:
            logger.info(f"Session {session_key[-5:]}: Finding related videos flag is TRUE. Starting YouTube processing...")

            # 5a. Extract Keywords (from processed transcript)
            keywords = extract_keywords_with_llm(processed_transcript)
            if not keywords:
                logger.warning(f"Session {session_key[-5:]}: Could not extract keywords via LLM, using first ~100 chars of transcript for YouTube search.")
                keywords = processed_transcript[:100] # Fallback search query

            # 5b. Process YouTube Videos (Search, Get Details, Fetch/Preprocess Transcripts)
            related_videos_data = process_related_youtube_videos(keywords) # Returns list of dicts

            if related_videos_data:
                youtube_transcripts = [v['transcript'] for v in related_videos_data] # Get processed transcripts
                # Prepare results for frontend display
                youtube_results_for_frontend = [{'title': v['title'], 'url': v['url']} for v in related_videos_data]

                # 5c. Chunk YouTube Transcripts
                all_youtube_chunks = []
                for i, yt_transcript in enumerate(youtube_transcripts):
                    logger.info(f"Session {session_key[-5:]}: Chunking YouTube transcript {i+1}/{len(youtube_transcripts)}...")
                    chunks = chunk_transcript_with_spacy(yt_transcript) # Use the same chunker
                    # Optional: Add metadata to chunks (e.g., source video title/URL)
                    # chunk_metadata = f"Source: {related_videos_data[i]['title']}"
                    # chunks_with_metadata = [f"{chunk}\n({chunk_metadata})" for chunk in chunks]
                    # all_youtube_chunks.extend(chunks_with_metadata)
                    all_youtube_chunks.extend(chunks)

                if all_youtube_chunks:
                    # 5d. Build FAISS Index for YouTube Chunks
                    logger.info(f"Session {session_key[-5:]}: Building FAISS index for {len(all_youtube_chunks)} YouTube chunks...")
                    # We only need embeddings_np to store in cache for later rebuilding
                    _, yt_embeddings_np = build_faiss_index(all_youtube_chunks)

                    if yt_embeddings_np is not None and yt_embeddings_np.size > 0:
                        # Store chunks and their embeddings in cache
                        cache.set(youtube_chunks_key, all_youtube_chunks, timeout=CACHE_TIMEOUT)
                        cache.set(youtube_embeddings_key, yt_embeddings_np, timeout=CACHE_TIMEOUT)
                        logger.info(f"Session {session_key[-5:]}: Stored {len(all_youtube_chunks)} YouTube chunks and embeddings in cache.")
                    else:
                        logger.warning(f"Session {session_key[-5:]}: Failed to generate or store embeddings for YouTube chunks.")
                else:
                    logger.warning(f"Session {session_key[-5:]}: No usable chunks generated from YouTube transcripts.")
            else:
                logger.info(f"Session {session_key[-5:]}: No related YouTube videos with usable transcripts found.")
        elif find_other_videos_flag and not YOUTUBE_API_KEY:
             logger.warning(f"Session {session_key[-5:]}: Find related videos flag is TRUE, but YouTube API Key is missing. Skipping YouTube search.")
        elif find_other_videos_flag and not processed_transcript:
              logger.warning(f"Session {session_key[-5:]}: Find related videos flag is TRUE, but base transcript is empty. Skipping YouTube search.")


        # --- 6. Handle Initial Question (If applicable) ---
        # This uses ONLY the base transcript, no RAG for the *initial* question here.
        initial_answer = None
        if initial_question and processed_transcript:
            logger.info(f"Session {session_key[-5:]}: Processing initial question using ONLY the base transcript...")

            # Construct Ollama Payload for Initial Question using BASE Transcript
            # Use the standard non-RAG prompt template for consistency
            initial_system_prompt = system_prompt_template_no_rag.format(
                context_text=processed_transcript
            )

            messages = [
                {"role": "system", "content": initial_system_prompt},
                {"role": "user", "content": initial_question}
                # No history for the very first question
            ]

            # Get FULL response for initial Q (accumulate from stream)
            # This part does NOT need the marker detection logic as it's the first turn.
            try:
                full_initial_response = ""
                response_stream = get_ollama_response_stream(messages)
                for chunk in response_stream:
                    decoded_chunk = chunk.decode('utf-8', errors='replace')
                    # Check for errors yielded by the stream helper
                    if "--- Error:" in decoded_chunk:
                        full_initial_response = decoded_chunk # Store the error message
                        break
                    full_initial_response += decoded_chunk

                # Process the complete response
                if "--- Error:" in full_initial_response:
                    # Extract error detail if possible
                    error_detail = full_initial_response.split('--- Error:')[1].split('---')[0].strip()
                    initial_answer = f"(AI processing failed for initial question: {error_detail})"
                    logger.error(f"Session {session_key[-5:]}: Ollama failed for initial Q: {error_detail}")
                # Check if response contains the permission prompt (Rule 3 was triggered)
                elif "PERMISSION_NEEDED\n" in full_initial_response:
                     # Extract the user-facing part
                     permission_question = full_initial_response.split("PERMISSION_NEEDED\n", 1)[1]
                     initial_answer = permission_question.strip()
                     # Set session state for the next turn
                     request.session['pending_general_knowledge_permission'] = True
                     request.session['original_question_for_general_knowledge'] = initial_question
                     request.session.save()
                     logger.info(f"Session {session_key[-5:]}: Initial question triggered permission prompt. State set.")
                elif full_initial_response:
                    initial_answer = full_initial_response.strip()
                    logger.info(f"Session {session_key[-5:]}: Generated initial answer using base transcript.")
                else:
                    logger.warning(f"Session {session_key[-5:]}: Ollama gave empty/no content for initial Q.")
                    initial_answer = "(AI did not provide an answer to the initial question)"

            except Exception as e:
                logger.error(f"Session {session_key[-5:]}: Ollama call failed for initial Q: {e}", exc_info=True)
                initial_answer = "(Error getting initial answer from AI)"

        elif initial_question: # Question asked but no transcript processed/available
            logger.warning(f"Session {session_key[-5:]}: Cannot answer initial question - transcript processing failed or yielded no content.")
            initial_answer = "(Cannot answer initial question as video processing yielded no usable text content)"

        # --- Final Response ---
        response_data = {
            'message': 'Video processed successfully. Ready for questions.',
             'title': video_title # Send back title if extracted from URL
             }
        if initial_answer:
            response_data['answer'] = initial_answer # Frontend uses this to start history
        if youtube_results_for_frontend:
            response_data['youtube_videos'] = youtube_results_for_frontend # Send YT links

        return JsonResponse(response_data)

    except Exception as e:
        # Catch-all for unexpected errors during the entire process
        logger.error(f"Unexpected error in upload_video (Session: {session_key[-5:] if session_key else 'Unknown'}): {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred during processing.'}, status=500)

    finally:
        # --- Cleanup Temporary Files ---
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.debug(f"Cleaned up temp audio file: {temp_audio_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp audio {temp_audio_path}: {e}")
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                logger.debug(f"Cleaned up temp video file: {uploaded_file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp video {uploaded_file_path}: {e}")
        logger.info(f"Session {session_key[-5:] if session_key else 'Unknown'}: Video upload processing finished.")


# --- Prompt Templates (Defined Here for Clarity) ---

# RAG Version - Updated Rule #3
system_prompt_template_rag = """You are an AI assistant answering questions about a primary
video and related YouTube videos. Remember the user's previous chat history.

Use the following sources:
1. **FULL ORIGINAL VIDEO TRANSCRIPT:** The main content. Prioritize this.
2. **RELEVANT EXCERPTS FROM RELATED YOUTUBE VIDEOS:** Additional context.

**RULES:**
1. **Base Specific Answers on Transcript:** Search the 'FULL VIDEO TRANSCRIPT' first for answers to specific factual questions.
2. **Answer Found:** If the answer IS in the transcript, provide it based **strictly** on the transcript text. Do not add outside information.
3. **Specific Answer NOT Found (CRITICAL RULE - FOLLOW EXACTLY):** If you are CERTAIN the answer to a specific factual question asked by the user is NOT in the 'FULL VIDEO TRANSCRIPT', you MUST follow these steps precisely:
    a. Your response MUST START *EXACTLY* with the line `PERMISSION_NEEDED` followed by a newline character (`\\n`).
    b. Immediately after the newline, you MUST provide *ONLY* the following sentence: 'This specific detail doesn't seem to be covered in the provided video transcript. Should I answer using my general knowledge? (Yes/No)'
    c. **CRITICALLY IMPORTANT:** Do NOT add *any* other words, phrases, introductions (like "However, I can provide..."), or the general knowledge answer itself in this response. Your entire output in this specific case must be ONLY the marker and the permission question sentence. Wait for the user's 'Yes' in the *next* turn before providing the general knowledge answer.
4. **General Knowledge Answer (After Permission):** ONLY if you previously triggered Rule #3 and the user has now replied 'Yes', answer the *original question* using your general knowledge, stating clearly that you are doing so.
5. **Summarization/Transcript Requests:** Summarize or provide transcripts based *only* on the provided text(s).
6. **Focus:** Focus on the user's LATEST question.
7. **General Topic Discussion:** You can discuss broader topics from the video using general knowledge, but clearly indicate when doing so (e.g., "Speaking more generally..."). **This does NOT override the strict requirements of Rule #3 for specific questions not found in the transcript.**

**FULL ORIGINAL VIDEO TRANSCRIPT:**
---
{original_transcript}
---
**RELEVANT EXCERPTS FROM RELATED YOUTUBE VIDEOS:**
---
{youtube_excerpts}
---
Now, answer the user's question below, following all rules meticulously:"""

# Non-RAG Version - Updated Rule #3
system_prompt_template_no_rag = """You are an AI assistant answering questions about the
video transcript provided below. Remember the user's previous chat history.

Use the following source:
1. **FULL VIDEO TRANSCRIPT:** The main content. Base your answers strictly on this.

**RULES:**
1. **Base Specific Answers on Transcript:** Search the 'FULL VIDEO TRANSCRIPT' first for answers to specific factual questions.
2. **Answer Found:** If the answer IS in the transcript, provide it based **strictly** on the transcript text. Do not add outside information.
3. **Specific Answer NOT Found (CRITICAL RULE - FOLLOW EXACTLY):** If you are CERTAIN the answer to a specific factual question asked by the user is NOT in the 'FULL VIDEO TRANSCRIPT', you MUST follow these steps precisely:
    a. Your response MUST START *EXACTLY* with the line `PERMISSION_NEEDED` followed by a newline character (`\\n`).
    b. Immediately after the newline, you MUST provide *ONLY* the following sentence: 'This specific detail doesn't seem to be covered in the provided video transcript. Should I answer using my general knowledge? (Yes/No)'
    c. **CRITICALLY IMPORTANT:** Do NOT add *any* other words, phrases, introductions (like "However, I can provide..."), or the general knowledge answer itself in this response. Your entire output in this specific case must be ONLY the marker and the permission question sentence. Wait for the user's 'Yes' in the *next* turn before providing the general knowledge answer.
4. **General Knowledge Answer (After Permission):** ONLY if you previously triggered Rule #3 and the user has now replied 'Yes', answer the *original question* using your general knowledge, stating clearly that you are doing so.
5. **Summarization/Transcript Requests:** Summarize or provide transcripts based *only* on the provided text.
6. **Focus:** Focus on the user's LATEST question.
7. **General Topic Discussion:** You can discuss broader topics from the video using general knowledge, but clearly indicate when doing so (e.g., "Speaking more generally..."). **This does NOT override the strict requirements of Rule #3 for specific questions not found in the transcript.**

**FULL VIDEO TRANSCRIPT:**
---
{context_text}
---
Now, answer the user's question below, following all rules meticulously:"""

# New Prompt for when General Knowledge is permitted
general_knowledge_prompt_template = """You are an AI assistant. The user previously asked a
question that could not be answered from the provided video transcript(s). The user has now given
you permission to answer using your general knowledge.

Remember the previous conversation history provided.

**User's Original Question (Answer this now using general knowledge):**
{original_question}

**Task:** Answer the user's original question clearly and comprehensively using your general
knowledge. Indicate that you are answering based on general knowledge since the information
wasn't in the video transcript.

Previous Conversation History is provided below. Use it for context but answer the original
question stated above.
"""


# View 3: Ask Question (Handles follow-up questions, RAG, and permission state)
@csrf_exempt
def ask_question(request):
    """
    Handles follow-up questions, potentially using RAG, and manages state
    for general knowledge permission.
    """
    if request.method != 'POST':
        # Return error via streaming response for consistency
        error_stream = iter([b"Error: Method not allowed."])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=405)

    session_key = request.session.session_key
    if not session_key:
        logger.warning("Ask question called without a valid session key.")
        error_stream = iter([b"Error: Your session has expired or is invalid. Please upload the video again."])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
        question = data.get('question', '').strip()
        client_history = data.get('history', []) # History sent by client (list of dicts)
        # Check find_other_videos flag sent from client with the question
        find_other_videos_flag = data.get('find_other_videos', False)
        logger.info(f"Session {session_key[-5:]}: Received question. Find YT Flag: {find_other_videos_flag}")
    except json.JSONDecodeError:
        error_stream = iter([b"Error: Invalid JSON payload."])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=400)
    except Exception as e:
        logger.error(f"Session {session_key[-5:]}: Error decoding request body: {e}", exc_info=True)
        error_stream = iter([b"Error: Failed to process request data."])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=400)

    if not question:
        error_stream = iter([b"Error: Question cannot be empty."])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=400)

    # --- Check Session State for Pending Permission ---
    permission_pending = request.session.get('pending_general_knowledge_permission', False)
    original_question_for_gk = request.session.get('original_question_for_general_knowledge', None)

    if permission_pending and original_question_for_gk:
        logger.info(f"Session {session_key[-5:]}: Permission for general knowledge is pending for question: '{original_question_for_gk}'")
        # Clear state immediately before processing the Yes/No
        request.session['pending_general_knowledge_permission'] = False
        request.session['original_question_for_general_knowledge'] = None
        # Important: Save session changes *now* in case of errors later
        request.session.save()

        if question.strip().lower() == 'yes':
            logger.info(f"Session {session_key[-5:]}: User responded YES. Answering original question ('{original_question_for_gk}') using general knowledge.")

            # Construct messages for general knowledge answer
            system_prompt_gk = general_knowledge_prompt_template.format(
                original_question=original_question_for_gk
            )
            print("gk")
            messages_for_ollama = [{"role": "system", "content": system_prompt_gk}]

            # Add limited history for context leading up to the permission request
            # Exclude the last AI perm-request and user's "Yes" for clarity? Optional.
            limited_history = client_history[-(MAX_HISTORY_TURNS * 2):]
            messages_for_ollama.extend(limited_history)
            # Don't append the user's "Yes" as current question; system prompt directs the AI

            try:
                response_stream = get_ollama_response_stream(messages_for_ollama)
                response = StreamingHttpResponse(response_stream, content_type="text/plain; charset=utf-8")
                response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response['Pragma'] = 'no-cache'
                response['Expires'] = '0'
                logger.info(f"Session {session_key[-5:]}: Streaming general knowledge response.")
                return response
            except Exception as e:
                logger.error(f"Session {session_key[-5:]}: Failed to initiate Ollama stream for GK: {e}", exc_info=True)
                error_stream = iter([b"\n\n--- Error: Failed to communicate with AI model for general knowledge answer ---"])
                return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=500)

        else: # User responded "No" or something else
            logger.info(f"Session {session_key[-5:]}: User did NOT respond YES to general knowledge.")
            ack_message = "Okay, I will not answer using my general knowledge for that question. How else can I help?"
            ack_stream = iter([ack_message.encode('utf-8')])
            response = StreamingHttpResponse(ack_stream, content_type="text/plain; charset=utf-8")
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            return response
            # End of permission handling block - execution stops here for this request.

    # --- If not handling a pending permission, proceed with normal question processing ---
    logger.info(f"Session {session_key[-5:]}: Processing question normally: '{question}'")

    # Check for essential models needed for subsequent steps
    if not EMBEDDING_MODEL or EMBEDDING_DIMENSION is None:
        error_message = "Error: Core components needed for answering are not loaded."
        logger.error(f"Session {session_key[-5:]}: {error_message}")
        error_stream = iter([error_message.encode('utf-8')])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=503)

    # --- Retrieve Context from Cache ---
    full_transcript_key = f"full_transcript_{session_key}"
    youtube_chunks_key = f"youtube_chunks_{session_key}"
    youtube_embeddings_key = f"youtube_embeddings_{session_key}"

    original_transcript = cache.get(full_transcript_key)
    # Check if original transcript exists in cache, critical for any answer
    if original_transcript is None: # Use 'is None' check for clarity
        logger.warning(f"Session {session_key[-5:]}: Original transcript not found in cache. Session might have expired.")
        no_context_message = "Error: The context for this video session seems to have expired. Please upload the video or provide the URL again."
        error_stream = iter([no_context_message.encode('utf-8')])
        # Use 404 Not Found or 410 Gone? 400 Bad Request might also fit if session invalid.
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=400)

    # --- Determine if RAG should be used ---
    youtube_chunks = None
    youtube_embeddings_np = None
    use_rag = False

    # Check only if the flag from the *current* request is true
    if find_other_videos_flag:
        youtube_chunks = cache.get(youtube_chunks_key)
        youtube_embeddings_np = cache.get(youtube_embeddings_key)
        if youtube_chunks and youtube_embeddings_np is not None and youtube_embeddings_np.size > 0:
            # Validate embedding dimension consistency
            if youtube_embeddings_np.shape[1] == EMBEDDING_DIMENSION:
                use_rag = True
                logger.info(f"Session {session_key[-5:]}: Using RAG. Found {len(youtube_chunks)} YouTube chunks and valid embeddings in cache.")
            else:
                logger.error(f"Session {session_key[-5:]}: RAG flag True, but YouTube embedding dimension mismatch! Expected {EMBEDDING_DIMENSION}, Got {youtube_embeddings_np.shape[1]}. Disabling RAG for this query.")
                use_rag = False
                # Optionally clear the bad cache entries?
                # cache.delete(youtube_chunks_key)
                # cache.delete(youtube_embeddings_key)
        else:
            logger.info(f"Session {session_key[-5:]}: RAG flag True, but no valid YouTube chunks/embeddings found in cache. Using original transcript only.")
            use_rag = False
    else:
         logger.info(f"Session {session_key[-5:]}: RAG flag is False. Using original transcript only.")
         use_rag = False


    # --- Prepare Context and Prompt ---
    relevant_youtube_context = ""
    system_prompt = ""
    messages_for_ollama = []

    # --- RAG Path (if use_rag is True) ---
    if use_rag:
        logger.info(f"Session {session_key[-5:]}: Performing RAG search...")
        try:
            # 1. Embed Question
            question_embedding = EMBEDDING_MODEL.encode([question])[0].astype('float32').reshape(1, -1)
            # faiss.normalize_L2(question_embedding) # Normalize if using IndexFlatIP

            # 2. Rebuild FAISS index from cached embeddings
            # Using L2 distance (Euclidean)
            index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            # If using cosine similarity:
            # index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            index.add(youtube_embeddings_np) # Add cached embeddings
            logger.debug(f"Session {session_key[-5:]}: FAISS index rebuilt with {index.ntotal} vectors for RAG search.")

            # 3. Search Index
            k = min(RAG_TOP_K, index.ntotal) # Determine number of neighbors to retrieve
            if k > 0:
                distances, indices = index.search(question_embedding, k)
                # Filter out invalid indices (-1) and ensure they are within bounds
                if indices.size > 0:
                     valid_indices = [i for i in indices[0] if 0 <= i < len(youtube_chunks)]
                     if valid_indices:
                         relevant_context_chunks = [youtube_chunks[i] for i in valid_indices]
                         relevant_youtube_context = "\n\n---\n\n".join(relevant_context_chunks)
                         logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} relevant YouTube chunks.")
                     else:
                         logger.warning(f"Session {session_key[-5:]}: RAG search returned indices out of bounds or invalid after filtering: {indices[0]}. No relevant chunks used.")
                else:
                    logger.info(f"Session {session_key[-5:]}: RAG search did not find any relevant YouTube chunks (indices: {indices}).")
            else:
                 logger.info(f"Session {session_key[-5:]}: Not enough vectors in index (k={k}) for RAG search.")

        except Exception as e:
            logger.error(f"Session {session_key[-5:]}: RAG search failed: {e}", exc_info=True)
            relevant_youtube_context = "(Error performing related video search)" # Inform LLM

        # Construct RAG System Prompt using the appropriate template
        context_text_youtube = relevant_youtube_context if relevant_youtube_context else "(No relevant excerpts found or search failed)"
        system_prompt = system_prompt_template_rag.format(
            original_transcript=(original_transcript or "(Original transcript unavailable)"),
            youtube_excerpts=context_text_youtube
        )

    # --- Non-RAG Path (use_rag is False) ---
    else:
        # Use the simpler prompt focusing only on the original transcript
        system_prompt = system_prompt_template_no_rag.format(
            context_text=(original_transcript or "(Transcript unavailable)")
        )

    # --- Construct Final Message List for Ollama ---
    messages_for_ollama = [{"role": "system", "content": system_prompt}]
    # Add limited history turns from client data
    limited_history = client_history[-(MAX_HISTORY_TURNS * 2):]
    messages_for_ollama.extend(limited_history)
    # Add the current user question
    messages_for_ollama.append({"role": "user", "content": question})

    # --- Stream Response with Marker Handling ---
    try:
        # Get the raw stream from Ollama
        raw_stream = get_ollama_response_stream(messages_for_ollama)

        # Define the generator function to wrap the raw stream and check for the marker
        def marker_checking_stream_generator(stream, current_question):
            permission_marker = b"PERMISSION_NEEDED\n"
            buffer = b""
            marker_found = False
            checked_marker = False # Flag to ensure we only check the very beginning

            for chunk in stream:
                if not checked_marker:
                    buffer += chunk
                    # Check if buffer starts with the marker
                    if buffer.startswith(permission_marker):
                        # Marker detected! Set session state
                        # Use request object from the outer scope
                        request.session['pending_general_knowledge_permission'] = True
                        request.session['original_question_for_general_knowledge'] = current_question
                        request.session.save() # Save session immediately
                        logger.info(f"Session {session_key[-5:]}: PERMISSION_NEEDED marker detected for question '{current_question}'. Session state set.")

                        # Yield the part of the buffer *after* the marker
                        remaining_chunk = buffer[len(permission_marker):]
                        if remaining_chunk:
                            yield remaining_chunk
                        marker_found = True
                        checked_marker = True # Stop checking for the marker

                    # If buffer is larger than marker and doesn't start with it, marker is not present at start
                    elif len(buffer) > len(permission_marker):
                        # Marker not found at the start, yield the whole buffer so far
                        yield buffer
                        buffer = b"" # Reset buffer as it's been yielded
                        checked_marker = True # Stop checking for the marker
                else:
                    # Marker state already determined, just yield subsequent chunks
                    yield chunk

            # After the loop, if we were still buffering waiting for enough data, yield any remainder
            if not checked_marker and buffer:
                 yield buffer

        # Create the response using the wrapper generator
        processed_stream = marker_checking_stream_generator(raw_stream, question)
        response = StreamingHttpResponse(processed_stream, content_type="text/plain; charset=utf-8")

        # Set cache control headers for the streaming response
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        logger.info(f"Session {session_key[-5:]}: Streaming response initiated (RAG Used: {use_rag}). Marker check active.")
        return response

    except Exception as e:
        logger.error(f"Session {session_key[-5:]}: Failed to initiate Ollama stream or marker check: {e}", exc_info=True)
        error_stream = iter([b"\n\n--- Error: Failed to communicate with AI model ---"])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=500)