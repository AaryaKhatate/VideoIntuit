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
import nltk
import torch
import yt_dlp
import ollama
import numpy as np # <-- Added
import faiss # <-- Added
import spacy
import requests # <-- Added

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound # <-- Added

# Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#--- YouTube API Key Handling ---
try:
    # Attempt to import from config.py first (recommended for local dev)
    from config import YOUTUBE_API_KEY
    logger.info("Successfully imported YOUTUBE_API_KEY from config.py")
except ImportError:
    logger.warning("Could not import YOUTUBE_API_KEY from config.py.")
    # Fallback to environment variable (recommended for production)
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", None)
    if YOUTUBE_API_KEY:
        logger.info("Found YOUTUBE_API_KEY in environment variables.")
    else:
        logger.error("CRITICAL: YOUTUBE_API_KEY not found in config.py or environment variables. YouTube search will fail.")
        # You might want to prevent the app from starting or disable features if the key is crucial
        # raise RuntimeError("YouTube API Key is missing.")
        YOUTUBE_API_KEY = None # Explicitly set to None if not found


#--- Global Model Loading (Load Once at Startup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

WHISPER_MODELS = {}
SPACY_NLP = None
EMBEDDING_MODEL = None
SPELL_CHECKER = None
EMBEDDING_DIMENSION = None

# Load Whisper Model
try:
    WHISPER_MODEL_NAME = getattr(settings, 'WHISPER_MODEL_NAME', "medium")
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

#--- Configuration ---
FFMPEG_COMMAND = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
CACHE_TIMEOUT = getattr(settings, 'CHAT_CACHE_TIMEOUT', 3600) # 1 hour cache
OLLAMA_MODEL = getattr(settings, 'OLLAMA_MODEL', 'llama3.2') # Your preferred Ollama model
CHUNK_SIZE = getattr(settings, 'TRANSCRIPT_CHUNK_SIZE', 300) # Smaller chunk size for RAG might be better
RAG_TOP_K = getattr(settings, 'RAG_TOP_K', 4) # Number of YouTube chunks to retrieve
MAX_HISTORY_TURNS = getattr(settings, 'MAX_HISTORY_TURNS', 10)
YT_INITIAL_SEARCH_RESULTS = getattr(settings, 'YT_INITIAL_SEARCH_RESULTS', 15) # How many YT videos to check initially
YT_TRANSCRIPT_LANGUAGES = getattr(settings, 'YT_TRANSCRIPT_LANGUAGES', ['en']) # Prioritize English transcripts

# --- Helper Functions ---

# --- Audio Extraction Helpers (Unchanged from original) ---
def extract_audio_from_file(video_source_path, target_audio_path):
    """Extracts audio from a local video file using FFmpeg."""
    logger.info(f"Extracting audio from file '{os.path.basename(video_source_path)}'...")
    command = [
        FFMPEG_COMMAND,
        "-i", video_source_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error",
        "-y",
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
    """
    Downloads video/audio, extracts audio, and tries to get the video title.
    Returns (success_boolean, title_string_or_none)
    """
    logger.info(f"Attempting to extract audio and title from URL: {video_url}")
    base_target = target_audio_path.rsplit('.', 1)[0]
    ffmpeg_location = getattr(settings, 'FFMPEG_COMMAND', 'ffmpeg')
    ffmpeg_location = ffmpeg_location if ffmpeg_location != 'ffmpeg' else None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{base_target}.%(ext)s',
        'noplaylist': True,
        'quiet': True, # Keep logs cleaner
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': {
            'extractaudio': ['-ar', '16000', '-ac', '1']
        },
        'ffmpeg_location': ffmpeg_location,
        'retries': 3,
        'socket_timeout': 60, # Increased timeout for potentially larger non-YT files
        'extract_flat': 'discard_in_playlist',
        'forcejson': False,
         # Try to get metadata without full download first if possible, might not always work
        'skip_download': False, # Ensure download happens
        'writethumbnail': False, # Don't need thumbnail
        'getfilename': False, # Don't just get filename
        'gettitle': True, # Explicitly ask for title
    }

    downloaded_correctly = False
    video_title = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Use extract_info to get metadata *and* trigger download if needed
            info_dict = ydl.extract_info(video_url, download=True) # Download=True ensures processing
            video_title = info_dict.get('title', None)
            logger.info(f"Extracted Title (if available): {video_title}")

        # Check if the specific target WAV file was created
        if os.path.exists(target_audio_path) and os.path.getsize(target_audio_path) > 0:
            downloaded_correctly = True
            logger.info(f"Audio download/extraction via yt-dlp successful: {target_audio_path}")
            return True, video_title # Return success and title
        else:
            # Check for alternatives and rename logic (as before)
            parent_dir = os.path.dirname(base_target) or '.'
            base_filename = os.path.basename(base_target)
            for f in os.listdir(parent_dir):
                if f.startswith(base_filename) and f.endswith('.wav') and os.path.exists(os.path.join(parent_dir, f)):
                    try:
                        os.rename(os.path.join(parent_dir, f), target_audio_path)
                        logger.info(f"Renamed intermediate file {f} to {target_audio_path}")
                        downloaded_correctly = True
                        return True, video_title # Return success and title
                    except OSError as rename_err:
                        logger.error(f"Failed to rename {f} to {target_audio_path}: {rename_err}")
                        break
            if not downloaded_correctly:
                 logger.error(f"yt-dlp finished, but target WAV not found: {target_audio_path}")
                 # Even if audio fails, we might have the title
                 return False, video_title # Indicate audio failure, but return title if found

    except yt_dlp.utils.DownloadError as e:
        # Check if it's because it's not a video/audio format yt-dlp recognizes
        if "Unsupported URL" in str(e) or "Unable to extract" in str(e):
             logger.warning(f"yt-dlp could not process URL (likely not video/audio): {video_url}. Error: {e}")
        else:
            logger.error(f"yt-dlp download error for {video_url}: {e}")
        return False, None # Indicate failure, no title
    except Exception as e:
        logger.error(f"Unexpected yt-dlp Error for {video_url}: {e}", exc_info=True)
        return False, None # Indicate failure, no title
    finally:
        # Cleanup logic (same as before)
        if not downloaded_correctly:
             try:
                 parent_dir = os.path.dirname(base_target) or '.'
                 base_filename = os.path.basename(base_target)
                 for f in os.listdir(parent_dir):
                     if f.startswith(base_filename) and f != os.path.basename(target_audio_path):
                         try:
                             full_path = os.path.join(parent_dir, f)
                             os.remove(full_path)
                             logger.warning(f"Cleaned up intermediate yt-dlp file: {full_path}")
                         except OSError: pass
             except Exception as cleanup_err:
                 logger.warning(f"Error during URL download cleanup: {cleanup_err}")

# --- Transcription Helper (Unchanged from original) ---
def transcribe_audio(audio_path):
    """Transcribes audio using the pre-loaded Faster Whisper model."""
    global WHISPER_MODEL_NAME
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
        # Added vad_filter=True for potentially better accuracy on long silences
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


#--- Text Preprocessing Helpers (Largely unchanged, ensure SPELL_CHECKER check works) ---
def remove_noise(text):
    """Removes common filler words."""
    text = re.sub(r'\b(um|uh|ah|er|hmm|uh-huh|um-hum)\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text) # Remove extra spaces
    return text.strip()

def remove_repeated_words(text):
    """Removes consecutive duplicate words."""
    return re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text, flags=re.IGNORECASE).strip()

def correct_spelling(text):
    """Corrects spelling using pyspellchecker and handles spacing around punctuation."""
    if not SPELL_CHECKER:
        logger.warning("SpellChecker not available, skipping spelling correction.")
        return text

    words_and_punct = re.findall(r"[\w'-]+|[.,!?;:]+|\S", text) # Keep contractions, handle punctuation
    corrected_tokens = []
    for token in words_and_punct:
        if re.fullmatch(r"[\w'-]+", token) and len(token) > 1:
            if not any(c.isupper() for c in token): # Avoid correcting potential acronyms/proper nouns
                corrected = SPELL_CHECKER.correction(token.lower())
                corrected_tokens.append(corrected if corrected and corrected != token.lower() else token)
            else:
                corrected_tokens.append(token) # Keep words with uppercase
        else:
            corrected_tokens.append(token) # Keep punctuation/symbols

    # Join tokens, carefully managing spaces
    processed_text = ""
    for i, token in enumerate(corrected_tokens):
        processed_text += token
        if i < len(corrected_tokens) - 1 and not re.fullmatch(r"[.,!?;:]", corrected_tokens[i+1]):
             # Add space unless next is punctuation
             # Also check if current token is not an opening bracket/quote needing no space after
             if not re.fullmatch(r"[(\[{'\"']", token):
                 processed_text += " "

    # Refine spacing around punctuation
    processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text) # Remove space BEFORE specific punctuation
    processed_text = re.sub(r'([.,!?;:])(?!\s)', r'\1 ', processed_text) # Add space AFTER punctuation if not followed by space
    # Handle potential space added after closing bracket/quote
    processed_text = re.sub(r"([)\]}'\"])\s", r"\1", processed_text)
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

    # Capitalize first letter of the whole text
    if processed:
         processed = processed[0].upper() + processed[1:]

    # Optional: Capitalize after sentence-ending punctuation (basic)
    processed = re.sub(r'([.!?])\s*(\w)', lambda m: m.group(1) + ' ' + m.group(2).upper(), processed)


    end_time = time.time()
    logger.info(f"Preprocessing finished in {end_time - start_time:.2f}s. Len: {original_length} -> {len(processed)}")
    logger.debug(f"Preprocessed Transcript (first 500 chars): {processed[:500]}...")
    return processed

# --- Chunking Helper (Uncommented and adapted) ---
def chunk_transcript_with_spacy(transcript, chunk_size=CHUNK_SIZE):
    """Splits the transcript into semantic chunks using spaCy sentences."""
    if not SPACY_NLP:
        logger.error("spaCy model not loaded. Cannot chunk transcript.")
        # Basic fallback: split by paragraphs or fixed length
        fallback_chunks = [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
        if not fallback_chunks:
            # If no paragraphs, split by approx length
            return [transcript[i:i+chunk_size].strip() for i in range(0, len(transcript), chunk_size) if transcript[i:i+chunk_size].strip()]
        return fallback_chunks

    if not transcript:
        return []

    logger.info(f"Chunking transcript using spaCy (Target Chunk Size: {chunk_size})...")
    start_time = time.time()
    # Process in batches if transcript is very large (e.g., > 1,000,000 chars)
    # For simplicity, processing all at once here. Ensure server has enough RAM.
    try:
        # Increase max_length if needed, default is 1,000,000
        # SPACY_NLP.max_length = len(transcript) + 100
        doc = SPACY_NLP(transcript)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ValueError as e:
         logger.error(f"spaCy max_length potentially exceeded: {e}. Trying paragraph splitting.")
         return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]
    except Exception as e:
        logger.error(f"spaCy processing failed: {e}. Falling back to paragraph splitting.")
        return [chunk.strip() for chunk in transcript.split('\n\n') if chunk.strip()]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence_len = len(sentence)
        current_chunk_len = len(current_chunk)

        if sentence_len > chunk_size:
            # If a single sentence is too long, add previous chunk and then the long sentence
            if current_chunk:
                 chunks.append(current_chunk)
            logger.warning(f"Sentence length ({sentence_len}) > CHUNK_SIZE ({chunk_size}). Adding as single chunk.")
            chunks.append(sentence)
            current_chunk = "" # Reset
        elif current_chunk_len == 0:
            # Start new chunk
            current_chunk = sentence
        elif current_chunk_len + sentence_len + 1 <= chunk_size: # +1 for space
            # Add sentence to current chunk if it fits
            current_chunk += " " + sentence
        else:
            # Current chunk is full, add it and start new chunk with current sentence
            chunks.append(current_chunk)
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    end_time = time.time()
    logger.info(f"Chunking complete in {end_time - start_time:.2f}s. Created {len(chunks)} chunks.")
    return chunks

# --- RAG Indexing Helper (Uncommented and adapted) ---
def build_faiss_index(transcript_chunks):
    """Builds a FAISS index from transcript chunks."""
    if not EMBEDDING_MODEL or not EMBEDDING_DIMENSION:
        logger.error("Embedding model/dimension not available. Cannot build FAISS index.")
        return None, None # Return None for both index and embeddings

    if not transcript_chunks:
        logger.warning("No transcript chunks provided to build FAISS index.")
        return None, None

    logger.info(f"Generating embeddings for {len(transcript_chunks)} chunks...")
    start_time = time.time()
    try:
        # Consider batch encoding if many chunks for efficiency
        embeddings = EMBEDDING_MODEL.encode(transcript_chunks, show_progress_bar=False, batch_size=128) # Adjust batch_size based on VRAM
        embeddings_np = np.array(embeddings).astype('float32')

        # Normalize embeddings for cosine similarity if using IndexFlatIP (Inner Product)
        # faiss.normalize_L2(embeddings_np) # Uncomment if using IndexFlatIP

        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s. Shape: {embeddings_np.shape}")

        logger.info("Building FAISS index (IndexFlatL2)...")
        # Using IndexFlatL2 (Euclidean distance). For cosine similarity, use IndexFlatIP and normalize embeddings.
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(embeddings_np)
        logger.info(f"FAISS index built successfully. Contains {index.ntotal} vectors.")

        # Return the built index AND the numpy embeddings (needed for potential cache rebuild)
        return index, embeddings_np
    except Exception as e:
        logger.error(f"Error building FAISS index or generating embeddings: {e}", exc_info=True)
        return None, None

# --- Ollama Streaming Helper (Unchanged) ---
def get_ollama_response_stream(messages_for_ollama):
    logger.info(f"Initiating Ollama stream with model '{OLLAMA_MODEL}', messages count: {len(messages_for_ollama)}")
    if messages_for_ollama:
         logger.debug(f"Ollama System Prompt (first 300 chars): {messages_for_ollama[0]['content'][:300]}...")
         logger.debug(f"Ollama User Message (last): {messages_for_ollama[-1]['content']}")
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
            if 'message' in part and 'content' in part['message']:
                yield part['message']['content'].encode('utf-8')
            if 'done' in part and part['done']:
                logger.info(f"Ollama stream finished. Reason: {part.get('done_reason', 'N/A')}")
                # You can log stats if needed: part.get('total_duration'), part.get('eval_count'), etc.

    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: Status {e.status_code}, Error: {e.error}")
        yield f"\n\n--- Error: Ollama API Error ({e.status_code}) ---".encode('utf-8')
    except Exception as e:
        logger.error(f"Error during Ollama stream: {e}", exc_info=True)
        yield f"\n\n--- Error: Could not communicate with AI model ---".encode('utf-8')

# --- Ollama Non-Streaming Helper (for keywords) ---
def call_llm(messages, model=OLLAMA_MODEL):
    logger.info(f"Calling Ollama model '{model}' (non-streaming) with {len(messages)} messages.")
    try:
        response = ollama.chat(model=model, messages=messages)
        logger.info(f"Ollama non-streaming response received.")
        return response['message']['content']
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error (non-streaming): Status {e.status_code}, Error: {e.error}")
        return f"Error: Ollama API Error ({e.status_code})"
    except Exception as e:
        logger.error(f"Error calling Ollama (non-streaming): {e}", exc_info=True)
        return f"Error: Could not communicate with AI model"

# --- Keyword Extraction Helper ---
def extract_keywords_with_llm(transcript, num_keywords=5):
    """Extracts keywords from transcript using Ollama."""
    if not transcript:
        return ""
    logger.info(f"Extracting {num_keywords} keywords from transcript using LLM...")
    prompt = f"""Extract the {num_keywords} most important and relevant keywords or keyphrases from the following video transcript.
Focus on specific topics, names, or concepts discussed. List the keywords separated by commas, without any introduction or explanation.

Transcript:
---
{transcript[:4000]}
---

Keywords:""" # Limit transcript length sent for keywords

    messages = [
        # No system prompt needed for this simple task
        {"role": "user", "content": prompt}
    ]
    keywords_string = call_llm(messages)

    if keywords_string.startswith("Error:") or not keywords_string:
        logger.error(f"Keyword extraction failed or returned empty: {keywords_string}")
        return "" # Return empty string on failure

    # Basic cleanup
    keywords_string = keywords_string.replace("Keywords:", "").strip()
    keywords_list = [kw.strip() for kw in keywords_string.split(',') if kw.strip()]

    logger.info(f"Keywords extracted: {keywords_list}")
    return ", ".join(keywords_list[:num_keywords]) # Return as comma-separated string


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
        'maxResults': initial_max_results,
        'key': YOUTUBE_API_KEY
    }
    try:
        response = requests.get(search_url, params=params, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error(f"YouTube search timed out for query: {query}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during YouTube search: {e}")
        return []
    except json.JSONDecodeError:
         logger.error("Failed to decode YouTube search response JSON.")
         return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during YouTube search: {e}", exc_info=True)
        return []

    if 'error' in data:
        logger.error(f"YouTube API Error (Search): {data['error']['message']}")
        return []

    video_ids = []
    for item in data.get('items', []):
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
            logger.error(f"Network error fetching YouTube video details: {e}")
            continue # Skip this batch
        except json.JSONDecodeError:
             logger.error("Failed to decode YouTube details response JSON.")
             continue # Skip this batch
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching YouTube video details: {e}", exc_info=True)
            continue # Skip this batch


        if 'error' in data:
            logger.error(f"YouTube API Error (Details): {data['error']['message']}")
            # Don't stop processing other batches if one fails
            continue

        for item in data.get('items', []):
            video_id = item['id']
            title = item.get('snippet', {}).get('title', 'N/A')
            # Use .get() with default value 0 in case stats are missing/private
            view_count = int(item.get('statistics', {}).get('viewCount', 0))
            # Like count can be hidden, default to 0 if missing
            like_count = int(item.get('statistics', {}).get('likeCount', -1)) # Use -1 to indicate potentially hidden
            url = f"https://www.youtube.com/watch?v={video_id}" # Correct YouTube URL

            video_details[video_id] = {
                'id': video_id,
                'title': title,
                'url': url,
                'view_count': view_count,
                'like_count': like_count
            }
    logger.info(f"Successfully fetched details for {len(video_details)} YouTube videos.")
    return video_details

def fetch_youtube_transcript(video_id):
    """Fetches the transcript for a single YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to find and fetch the English transcript specifically
        transcript = transcript_list.find_generated_transcript(YT_TRANSCRIPT_LANGUAGES)
        # Alternative: fetch manually created transcript if generated isn't found
        # transcript = transcript_list.find_manually_created_transcript(YT_TRANSCRIPT_LANGUAGES)

        full_transcript = ' '.join([t['text'] for t in transcript.fetch()])
        return full_transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.debug(f"Transcript not available or disabled for YouTube video {video_id}.")
        return None
    except Exception as e:
        # Log other potential errors like network issues during fetch
        logger.warning(f"Error fetching YouTube transcript for {video_id}: {str(e)}")
        return None


def process_related_youtube_videos(query):
    """Orchestrates the YouTube search, detail fetching, transcript retrieval, and sorting."""
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
        return []

    logger.info("Fetching transcripts for potential YouTube videos...")
    valid_videos_with_transcripts = []
    checked_count = 0
    total_to_check = len(video_details)
    for video_id, details in video_details.items():
        checked_count += 1
        logger.debug(f"Checking YouTube video {checked_count}/{total_to_check} (ID: {video_id})...")
        transcript = fetch_youtube_transcript(video_id)
        if transcript:
            # Preprocess the YouTube transcript too
            processed_yt_transcript = preprocess_transcript(transcript)
            if processed_yt_transcript: # Ensure preprocessing didn't result in empty string
                 details['transcript'] = processed_yt_transcript
                 valid_videos_with_transcripts.append(details)
            else:
                 logger.debug(f"YouTube transcript for {video_id} became empty after preprocessing.")
        # Add a small delay to avoid hitting API limits too quickly if necessary
        # time.sleep(0.1)

    logger.info(f"Found {len(valid_videos_with_transcripts)} YouTube videos with available & non-empty transcripts.")

    if not valid_videos_with_transcripts:
        logger.info("No relevant YouTube videos found with usable transcripts.")
        return []

    # Sort by View Count (descending)
    sorted_videos = sorted(valid_videos_with_transcripts, key=lambda x: x['view_count'], reverse=True)

    # Select Top N (default RAG_TOP_K=4)
    final_selection = sorted_videos[:RAG_TOP_K]
    logger.info(f"Selected top {len(final_selection)} YouTube videos based on view count.")

    # Return the list of selected video dictionaries (id, title, url, transcript, view_count, like_count)
    return final_selection


# --- Django Views ---

# View 1: Render Index Page (Unchanged)
def index(request):
    """Renders the main chat page (index.html)."""
    # Basic model check
    models_ready = all([
        WHISPER_MODELS,
        # SPACY_NLP, # Only critical if chunking must always succeed
        EMBEDDING_MODEL, # Critical if RAG is potentially used
        EMBEDDING_DIMENSION is not None
    ])
    if not models_ready:
        logger.critical("One or more critical models (Whisper/Embedding) failed to load. Features may be limited.")
    return render(request, 'index.html')

# View 2: Upload Video (Handles initial processing, optional YT search/RAG setup)
@csrf_exempt
def upload_video(request):
    """
    Handles POST video/URL, processes, gets transcript, optionally finds related
    YouTube videos and sets up RAG context in cache, clears old history cache,
    and optionally answers an initial question.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    # Essential models check
    if not all([WHISPER_MODELS, EMBEDDING_MODEL]): # SPACY less critical now
        logger.error("Cannot process upload: Essential models (Whisper/Embedding) not loaded.")
        return JsonResponse({'error': 'Server is not ready. Models not loaded.'}, status=503)

    video_source = None
    source_type = None
    uploaded_file_path = None
    temp_audio_path = None
    session_key = None
    initial_question = None
    processed_transcript = None
    youtube_results = [] # To store YT video details {title, url} for frontend
    find_other_videos = False # Default

    try:
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key
        logger.info(f"Session {session_key[-5:]}: Starting video upload processing...")

        # --- Input Handling (File or URL + Flags) ---
        if request.FILES.get('videoFile'):
            source_type = 'file'
            video_file = request.FILES['videoFile']
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
                for chunk in video_file.chunks(): tmp_video.write(chunk)
                uploaded_file_path = tmp_video.name
            video_source = uploaded_file_path
            # Get other form data
            initial_question = request.POST.get('question', '').strip()
            find_other_videos = request.POST.get('find_other_videos', 'false').lower() == 'true' # Get flag from form
            logger.info(f"Session {session_key[-5:]}: Processing uploaded file '{video_file.name}'. Initial Q: {'Yes' if initial_question else 'No'}. Find YT: {find_other_videos}")

        elif request.content_type == 'application/json' and request.body:
            try:
                data = json.loads(request.body)
                if data.get('videoUrl'):
                    video_source = data['videoUrl']
                    source_type = 'url'
                    initial_question = data.get('question', '').strip()
                    find_other_videos = data.get('find_other_videos', False) # Get flag from JSON
                    logger.info(f"Session {session_key[-5:]}: Processing URL '{video_source}'. Initial Q: {'Yes' if initial_question else 'No'}. Find YT: {find_other_videos}")
                else:
                    return JsonResponse({'error': 'Missing videoUrl in JSON payload'}, status=400)
            except json.JSONDecodeError:
                logger.warning(f"Session {session_key[-5:]}: Invalid JSON payload received.")
                return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        else:
             return JsonResponse({'error': 'No video file or URL JSON payload provided'}, status=400)

        # --- Processing Steps ---
        # 1. Extract Audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_f:
            temp_audio_path = tmp_audio_f.name
        extraction_func = extract_audio_from_file if source_type == 'file' else extract_audio_from_video_url
        if not extraction_func(video_source, temp_audio_path):
            return JsonResponse({'error': 'Audio extraction failed'}, status=500)

        # 2. Transcribe Audio
        raw_transcript, trans_error = transcribe_audio(temp_audio_path)
        if trans_error: return JsonResponse({'error': trans_error}, status=500)
        if not raw_transcript:
             return JsonResponse({'error': 'Transcription resulted in empty text.'}, status=500)


        # 3. Preprocess Transcript
        processed_transcript = preprocess_transcript(raw_transcript)
        if not processed_transcript:
             logger.warning(f"Session {session_key[-5:]}: Preprocessing resulted in empty transcript.")
             # Decide if this is an error or just proceed without context
             # return JsonResponse({'error': 'Preprocessing resulted in empty text.'}, status=500)


        # 4. Store Base Transcript & Clear History
        full_transcript_key = f"full_transcript_{session_key}"
        history_key = f"conversation_history_{session_key}"
        youtube_chunks_key = f"youtube_chunks_{session_key}" # New cache key
        youtube_embeddings_key = f"youtube_embeddings_{session_key}" # New cache key

        cache.set(full_transcript_key, processed_transcript, timeout=CACHE_TIMEOUT)
        cache.delete(history_key) # Clear old history
        # Clear potential old YT data
        cache.delete(youtube_chunks_key)
        cache.delete(youtube_embeddings_key)

        logger.info(f"Session {session_key[-5:]}: Stored base transcript ({len(processed_transcript or '')} chars). Cleared old cache.")

        # 5. --- Optional: YouTube Search & RAG Setup ---
        if find_other_videos and YOUTUBE_API_KEY and processed_transcript:
            logger.info(f"Session {session_key[-5:]}: Finding related videos flag is TRUE. Starting YouTube processing...")
            # 5a. Extract Keywords
            keywords = extract_keywords_with_llm(processed_transcript)
            if not keywords:
                logger.warning(f"Session {session_key[-5:]}: Could not extract keywords, using first ~100 chars of transcript for YouTube search.")
                keywords = processed_transcript[:100] # Fallback

            # 5b. Process YouTube Videos
            related_videos = process_related_youtube_videos(keywords) # Returns list of dicts

            if related_videos:
                youtube_transcripts = [v['transcript'] for v in related_videos]
                youtube_results = [{'title': v['title'], 'url': v['url']} for v in related_videos] # For frontend

                # 5c. Chunk YouTube Transcripts
                all_youtube_chunks = []
                for i, yt_transcript in enumerate(youtube_transcripts):
                    logger.info(f"Session {session_key[-5:]}: Chunking YouTube transcript {i+1}/{len(youtube_transcripts)}...")
                    chunks = chunk_transcript_with_spacy(yt_transcript)
                    # Optional: Add metadata to chunks (e.g., source video title/URL) if needed later
                    all_youtube_chunks.extend(chunks)

                if all_youtube_chunks:
                    # 5d. Build FAISS Index for YouTube Chunks
                    logger.info(f"Session {session_key[-5:]}: Building FAISS index for {len(all_youtube_chunks)} YouTube chunks...")
                    _, yt_embeddings_np = build_faiss_index(all_youtube_chunks) # We only need embeddings for cache

                    if yt_embeddings_np is not None and yt_embeddings_np.size > 0:
                        # Store chunks and their embeddings in cache
                        cache.set(youtube_chunks_key, all_youtube_chunks, timeout=CACHE_TIMEOUT)
                        cache.set(youtube_embeddings_key, yt_embeddings_np, timeout=CACHE_TIMEOUT)
                        logger.info(f"Session {session_key[-5:]}: Stored {len(all_youtube_chunks)} YouTube chunks and embeddings in cache.")
                    else:
                        logger.warning(f"Session {session_key[-5:]}: Failed to generate embeddings for YouTube chunks.")
                else:
                     logger.warning(f"Session {session_key[-5:]}: No usable chunks generated from YouTube transcripts.")
            else:
                 logger.info(f"Session {session_key[-5:]}: No related YouTube videos with transcripts found.")

        # --- 6. Handle Initial Question (If applicable, using ONLY the base transcript for now) ---
        # RAG for the *initial* question adds complexity. We'll keep it simple:
        # The initial Q uses only the main video's transcript.
        # Subsequent questions will use RAG if the flag was set.
        initial_answer = None
        if initial_question and processed_transcript:
            logger.info(f"Session {session_key[-5:]}: Processing initial question using ONLY the base transcript...")

            # Construct Ollama Payload for Initial Question using BASE Transcript
            initial_system_prompt_template = """You are an AI assistant answering questions about the video transcript provided below. Base your answers strictly on the information within the 'FULL VIDEO TRANSCRIPT'. Do not use outside knowledge unless specifically asked or for general discussion.

            **CRITICAL RULE:** If the user asks a specific question and the answer is NOT found in the 'FULL VIDEO TRANSCRIPT', your *entire response* must be *only* the following sentence: "This specific detail doesn't seem to be covered in the provided video transcript. Should I answer using my general knowledge? (Yes/No)". Do NOT provide the general knowledge answer unless the user explicitly agrees in a follow-up message.

            FULL VIDEO TRANSCRIPT:
            ---
            {context_text}
            ---

            Answer the user's question below based ONLY on the transcript:"""
            context_text = processed_transcript
            initial_system_prompt = initial_system_prompt_template.format(context_text=context_text)
            messages = [
                {"role": "system", "content": initial_system_prompt},
                {"role": "user", "content": initial_question}
            ]

            # Get FULL response for initial Q (accumulate from stream)
            try:
                full_initial_response = ""
                response_stream = get_ollama_response_stream(messages)
                for chunk in response_stream:
                    decoded_chunk = chunk.decode('utf-8', errors='replace')
                    if "--- Error:" in decoded_chunk: # Catch errors from the stream helper
                         full_initial_response = decoded_chunk # Store the error message
                         break
                    full_initial_response += decoded_chunk

                if "--- Error:" in full_initial_response:
                    error_detail = full_initial_response.split('--- Error:')[1].split('---')[0].strip()
                    initial_answer = f"(AI processing failed for initial question: {error_detail})"
                    logger.error(f"Session {session_key[-5:]}: Ollama failed for initial Q: {error_detail}")
                elif full_initial_response and "(AI returned no content)" not in full_initial_response:
                    initial_answer = full_initial_response.strip()
                    logger.info(f"Session {session_key[-5:]}: Generated initial answer using base transcript.")
                else:
                    logger.warning(f"Session {session_key[-5:]}: Ollama gave empty/no content for initial Q.")
                    initial_answer = "(AI did not provide an answer to the initial question)"
            except Exception as e:
                logger.error(f"Session {session_key[-5:]}: Ollama call failed for initial Q: {e}", exc_info=True)
                initial_answer = "(Error getting initial answer from AI)"

        elif initial_question: # Question asked but no transcript processed
            logger.warning(f"Session {session_key[-5:]}: Cannot answer initial question - transcript processing failed or yielded no content.")
            initial_answer = "(Cannot answer initial question as video processing yielded no usable text content)"

        # --- Final Response ---
        response_data = {'message': 'Video processed successfully. Ready for questions.'}
        if initial_answer:
            response_data['answer'] = initial_answer # Frontend will use this to start history
        if youtube_results:
            response_data['youtube_videos'] = youtube_results # Send YT links to frontend

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
    """Handles follow-up questions, potentially using RAG with YouTube context."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    session_key = request.session.session_key
    if not session_key:
        # This might happen if the session expired or cookies are disabled client-side
        logger.warning("Ask question called without a valid session key.")
        return JsonResponse({'error': 'Your session has expired or is invalid. Please upload the video again.'}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
        question = data.get('question', '').strip()
        client_history = data.get('history', [])
        find_other_videos = data.get('find_other_videos', False) # Get flag from client

        logger.info(f"Session {session_key[-5:]}: Received question. Find YT Flag: {find_other_videos}")

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error decoding request body or getting session key: {e}", exc_info=True)
        return JsonResponse({'error': 'Failed to process request data.'}, status=400)


    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    # Check for essential components
    if not EMBEDDING_MODEL or EMBEDDING_DIMENSION is None:
         error_message = "Embedding models are not loaded. RAG functionality is disabled."
         logger.error(f"Session {session_key[-5:]}: {error_message}")
         return StreamingHttpResponse(iter([error_message.encode('utf-8')]),
                                       content_type="text/plain; charset=utf-8", status=503)

    # --- Retrieve Context from Cache ---
    full_transcript_key = f"full_transcript_{session_key}"
    youtube_chunks_key = f"youtube_chunks_{session_key}"
    youtube_embeddings_key = f"youtube_embeddings_{session_key}"

    original_transcript = cache.get(full_transcript_key)
    youtube_chunks = None
    youtube_embeddings_np = None
    use_rag = False

    if not original_transcript:
        logger.warning(f"Session {session_key[-5:]}: Original transcript not found in cache for session.")
        no_context_message = "It seems the context for this video is no longer available (it might have expired). Please upload the video or provide the URL again."
        return StreamingHttpResponse(iter([no_context_message.encode('utf-8')]),
                                   content_type="text/plain; charset=utf-8", status=200)

    # Check if RAG should be used and if data is available
    if find_other_videos:
        youtube_chunks = cache.get(youtube_chunks_key)
        youtube_embeddings_np = cache.get(youtube_embeddings_key)
        if youtube_chunks and youtube_embeddings_np is not None and youtube_embeddings_np.size > 0:
             # Validate embeddings dimension
             if youtube_embeddings_np.shape[1] == EMBEDDING_DIMENSION:
                 use_rag = True
                 logger.info(f"Session {session_key[-5:]}: Found valid YouTube chunks ({len(youtube_chunks)}) and embeddings for RAG.")
             else:
                  logger.error(f"Session {session_key[-5:]}: YouTube embedding dimension mismatch in cache! Expected {EMBEDDING_DIMENSION}, Got {youtube_embeddings_np.shape[1]}. Disabling RAG.")
                  use_rag = False
                  # Optionally clear the bad cache entries
                  # cache.delete(youtube_chunks_key)
                  # cache.delete(youtube_embeddings_key)
        else:
            logger.info(f"Session {session_key[-5:]}: Find other videos flag is True, but no valid YouTube chunks/embeddings found in cache. Using original transcript only.")
            use_rag = False

    # --- Prepare Context and Prompt ---
    relevant_youtube_context = ""
    system_prompt = ""
    messages_for_ollama = []

    # --- RAG Path (if flag is set AND YT data is available) ---
    if use_rag:
        logger.info(f"Session {session_key[-5:]}: Performing RAG search...")
        try:
            # 1. Embed Question
            question_embedding = EMBEDDING_MODEL.encode([question])[0].astype('float32').reshape(1, -1)
            # faiss.normalize_L2(question_embedding) # Normalize if using IndexFlatIP

            # 2. Rebuild FAISS index from cached embeddings
            index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            # index = faiss.IndexFlatIP(EMBEDDING_DIMENSION) # If using cosine similarity
            index.add(youtube_embeddings_np)
            logger.debug(f"Session {session_key[-5:]}: FAISS index rebuilt with {index.ntotal} vectors for RAG.")

            # 3. Search Index
            k = min(RAG_TOP_K, index.ntotal)
            if k > 0:
                distances, indices = index.search(question_embedding, k)
                if indices.size > 0 and np.all(indices[0] != -1): # Check if search returned valid indices
                    # Ensure indices are within the bounds of the retrieved chunks
                    valid_indices = [i for i in indices[0] if 0 <= i < len(youtube_chunks)]
                    if valid_indices:
                        relevant_context_chunks = [youtube_chunks[i] for i in valid_indices]
                        relevant_youtube_context = "\n\n---\n\n".join(relevant_context_chunks)
                        logger.info(f"Session {session_key[-5:]}: RAG found {len(relevant_context_chunks)} relevant YouTube chunks.")
                    else:
                        logger.warning(f"Session {session_key[-5:]}: RAG search returned indices out of bounds or became invalid: {indices[0]}.")
                else:
                     logger.info(f"Session {session_key[-5:]}: RAG search did not find any relevant YouTube chunks (indices: {indices}).")
            else:
                 logger.info(f"Session {session_key[-5:]}: Not enough vectors in index (or k=0) for RAG search.")

        except Exception as e:
            logger.error(f"Session {session_key[-5:]}: RAG search failed: {e}", exc_info=True)
            relevant_youtube_context = "(Error performing related video search)" # Inform LLM

        # Construct RAG System Prompt
        system_prompt_template_rag = """You are an AI assistant answering questions about a primary video and related YouTube videos.

        Use the following sources:
        1.  **FULL ORIGINAL VIDEO TRANSCRIPT:** The main content. Prioritize this for questions directly about the main video.
        2.  **RELEVANT EXCERPTS FROM RELATED YOUTUBE VIDEOS:** Additional context from similar videos found online.

        **RULES:**
        1.  **Base Answers on Provided Text:** Answer strictly based on the text provided in the Original Transcript and also refer the YouTube Excerpts.
        2.  **Cite Your Source:** When using information from the YouTube excerpts, clearly state it (e.g., "In a related video, it mentions...", "According to the excerpts from related videos...").
        3.  **Prioritize Original:** If the answer exists in both, prefer the Original Transcript's information unless the user asks for comparison or external perspectives.
        4.  **Answer Not Found:** If, after searching *both* sources, the answer to a specific question is NOT found, your *entire response* must be *only* the following sentence: "This specific detail doesn't seem to be covered in the provided video transcripts (original or related). Should I answer using my general knowledge? (Yes/No)". Do NOT provide the general knowledge answer unless the user explicitly agrees in a follow-up message.
        5.  **General Discussion:** You *can* use general knowledge for broader topic discussions *inspired* by the transcripts, but clearly distinguish this from transcript-based answers (e.g., "Speaking more generally about [topic]..."). Rule #4 still applies strictly for specific factual questions about the video content.
        6.  **Focus:** Address the user's latest question.

        **FULL ORIGINAL VIDEO TRANSCRIPT:**
        ---
        {original_transcript}
        ---

        **RELEVANT EXCERPTS FROM RELATED YOUTUBE VIDEOS:**
        ---
        {youtube_excerpts}
        ---

        Answer the user's question below, following all rules:"""

        context_text_youtube = relevant_youtube_context if relevant_youtube_context else "(No relevant excerpts found or search failed)"
        system_prompt = system_prompt_template_rag.format(
            original_transcript=(original_transcript or "(Original transcript unavailable)"),
            youtube_excerpts=context_text_youtube
        )

    # --- Non-RAG Path (Flag is false OR YT data is missing) ---
    else:
        # Use the simpler prompt focusing only on the original transcript
         system_prompt_template_no_rag = """You are an AI assistant answering questions about the video transcript provided below. Base your answers strictly on the information within the 'FULL VIDEO TRANSCRIPT'. Do not use outside knowledge unless specifically asked or for general discussion.

         **CRITICAL RULE:** If the user asks a specific question and the answer is NOT found in the 'FULL VIDEO TRANSCRIPT', your *entire response* must be *only* the following sentence: "This specific detail doesn't seem to be covered in the provided video transcript. Should I answer using my general knowledge? (Yes/No)". Do NOT provide the general knowledge answer unless the user explicitly agrees in a follow-up message.

         FULL VIDEO TRANSCRIPT:
         ---
         {context_text}
         ---

         Answer the user's question below based ONLY on the transcript:"""
         system_prompt = system_prompt_template_no_rag.format(
             context_text=(original_transcript or "(Transcript unavailable)")
         )


    # --- Construct Final Message List ---
    messages_for_ollama = [{"role": "system", "content": system_prompt}]
    # Limit history turns based on settings
    limited_history = client_history[-(MAX_HISTORY_TURNS * 2):]
    messages_for_ollama.extend(limited_history)
    messages_for_ollama.append({"role": "user", "content": question})

    # --- Stream Response ---
    try:
        response_stream = get_ollama_response_stream(messages_for_ollama)
        response = StreamingHttpResponse(response_stream, content_type="text/plain; charset=utf-8")
        # Set headers to prevent caching of the stream
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        logger.info(f"Session {session_key[-5:]}: Streaming response initiated (RAG Used: {use_rag}).")
        return response
    except Exception as e:
        logger.error(f"Session {session_key[-5:]}: Failed to initiate Ollama stream: {e}", exc_info=True)
        error_stream = iter([b"\n\n--- Error: Failed to communicate with AI model ---"])
        return StreamingHttpResponse(error_stream, content_type="text/plain; charset=utf-8", status=500)