import time
import os
import subprocess
import nltk
import torch
import yt_dlp
import ollama
import re
import spacy
from faster_whisper import WhisperModel
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Download nltk resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for spaCy...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Path to FFmpeg binary (Update as needed)
FFMPEG_PATH = r"C:\Users\Yashraj Patil\Downloads\ffmpeg-2025-01-30-git-1911a6ec26-full_build\ffmpeg-2025-01-30-git-1911a6ec26-full_build\bin\ffmpeg.exe"

# Initialize SpellChecker
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

# Initialize embedding model
embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)
embedding_dimension = embedding_model.get_sentence_embedding_dimension()

# Global variables for FAISS index and transcript chunks
faiss_index = None
transcript_chunks = []
CHUNK_SIZE = 512 # You can adjust this value

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
    print("Extracting audio from video...")
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

def remove_noise(text):
    """A more comprehensive noise removal function."""
    text = re.sub(r'\b(um|uh|ah|er|hmm|hmmm)\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(like|you know|so|well)\b\s*', '', text, flags=re.IGNORECASE)
    return text

def remove_repeated_words(text):
    """Removes consecutive duplicate words."""
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

spell = SpellChecker()

def correct_spelling(text):
    """Corrects spelling and ensures consistent spacing after sentence-ending punctuation."""
    corrected_tokens = []
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    for token in tokens:
        if re.match(r"[\w']+", token):
            corrected_word = spell.correction(token)
            corrected_tokens.append(corrected_word if corrected_word is not None else token)
        else:
            corrected_tokens.append(token)

    processed_text = " ".join(corrected_tokens)
    # Ensure exactly one space after '.', '?', '!'
    processed_text = re.sub(r"([.?!])\s*", r"\1 ", processed_text)
    processed_text = re.sub(r"\s+([.?!])", r"\1 ", processed_text) # Remove space before
    processed_text = re.sub(r"([.?!])\s*$", r"\1", processed_text) # Remove trailing space

    # Ensure space after comma
    processed_text = re.sub(r"(,)\s*", r"\1 ", processed_text)
    processed_text = re.sub(r"\s+(,)", r"\1 ", processed_text) # Remove space before

    return processed_text.strip()

def restore_punctuation(text):
    """A very basic punctuation restoration - might need improvement based on context."""
    text = re.sub(r'\s*([.,?!])', r'\1', text) # Remove spaces before punctuation
    text = re.sub(r'([a-zA-Z0-9])([A-Z])', r'\1. \2', text) # Basic sentence splitting
    return text.capitalize() # Capitalize the first word

def preprocess_transcript(transcript):
    """Applies preprocessing steps to the transcript and prints after each."""
    print("Preprocessing transcript...")

    #print("\n**Initial Transcript (Lowercase):**")
    transcript = transcript.lower() # Lowercase for consistency
    #print(transcript)

    #print("\n**After remove_noise:**")
    transcript = remove_noise(transcript)
    #print(transcript)

    #print("\n**After remove_repeated_words:**")
    transcript = remove_repeated_words(transcript)
    #print(transcript)

    #print("\n**After correct_spelling:**")
    transcript = correct_spelling(transcript)
    #print(transcript)

    return transcript

def transcribe_audio(audio_path):
    """Transcribes audio using Faster Whisper and prepares for vectorization."""
    print("Transcribing with Faster Whisper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("large", device=device, compute_type="float16" if device == "cuda" else "float32")
    segments, info = model.transcribe(audio_path, beam_size=5)
    detected_language = getattr(info, "language", "en")
    print(f"Detected Language: {detected_language.upper()}")
    model_size = "medium" if detected_language == "en" else "large"
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "float32")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    raw_transcript = " ".join([segment.text for segment in segments])
    return raw_transcript

def chunk_transcript_with_spacy(transcript, chunk_size=CHUNK_SIZE):
    """Splits the transcript into sentences using spaCy and then into chunks."""
    doc = nlp(transcript)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_and_build_faiss_index(transcript):
    global faiss_index, transcript_chunks
    processed_transcript = preprocess_transcript(transcript)
    print("\n**Processed Transcript (Before Chunking):**\n", processed_transcript)
    transcript_chunks = chunk_transcript_with_spacy(processed_transcript) # Use spaCy for chunking

    print(f"\n**Created {len(transcript_chunks)} semantic chunks.**")
    for i, chunk in enumerate(transcript_chunks):
        print(f"Chunk {i}: {chunk[:100]}...")

    embeddings = embedding_model.encode(transcript_chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    faiss_index = index
    print(f"Built FAISS index with {len(transcript_chunks)} chunks.")

# Continue to the second part for the `answer_question` and `main` functions.
def answer_question(conversation_history):
    global faiss_index, transcript_chunks
    model_name = "llama3.2"

    while True:
        question = input("\nEnter another question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        conversation_history.append({"role": "user", "content": question})

        relevant_context = ""
        if faiss_index is not None and transcript_chunks:
            question_embedding = embedding_model.encode([question])[0].astype('float32').reshape(1, -1)
            k = 3
            distances, indices = faiss_index.search(question_embedding, k)

            if indices.any():
                relevant_context_chunks = [transcript_chunks[i] for i in indices[0]]
                relevant_context = "\n".join(relevant_context_chunks)

                print("\n**Relevant Transcript Chunks (Used for Context):**")
                for i, chunk in enumerate(relevant_context_chunks):
                    print(f"Chunk {indices[0][i]}:\n{chunk}\n---")

        formatted_history = "\n".join([f"{item['role']}: {item['content']}" for item in conversation_history[:-1]])

        prompt = f"""You are an AI assistant tasked with answering questions based on the following video transcript excerpts:\n\n{relevant_context}\n\nConsidering the previous conversation:\n\n{formatted_history}\n\nIdentify if the current question is directly related to the topics discussed in the video transcript.\n\nIf the question is a completely new topic and not related to the video content, respond by first stating:\n"This question does not appear to be directly related to the video content."\nThen, immediately ask the user:\n"Would you like me to try and answer it using my general knowledge? (yes/no)"\n\nIf the user indicates 'yes' or a similar positive response in their next turn, then answer the original question using your general knowledge, considering the conversation history but without mentioning the video content's irrelevance.\n\nOtherwise, if the question is related to the video content, answer it based *only* on the provided video transcript excerpts if possible. If the answer cannot be found in the excerpts, truthfully state that it is not mentioned in the video.\n\nCurrent Question: {question}"""

        messages_for_ollama = [{"role": "user", "content": prompt}]

        try:
            stream = ollama.chat(
                model=model_name,
                messages=messages_for_ollama,
                stream=True,
            )
            print("\n**Ollama Answer:**\n")
            response_content = ""
            for part in stream:
                response_content += part['message']['content']
                print(part['message']['content'], end='', flush=True)
            conversation_history.append({"role": "assistant", "content": response_content})

            # Handle the general knowledge follow-up
            if "Would you like me to try and answer it using my general knowledge? (yes/no)" in response_content:
                general_knowledge_choice = input().strip().lower()
                conversation_history.append({"role": "user", "content": general_knowledge_choice})
                if general_knowledge_choice in ["yes", "y", "sure", "ok", "please"]:
                    general_knowledge_prompt = f"Answer the question '{question}' using your general knowledge, considering the previous conversation."
                    messages_for_ollama_general = [{"role": "user", "content": general_knowledge_prompt}] # Start a new prompt for general knowledge
                    print("\n**Ollama (General Knowledge Answer):**\n")
                    stream_general = ollama.chat(
                        model=model_name,
                        messages=messages_for_ollama_general,
                        stream=True,
                    )
                    general_response_content = ""
                    for part_general in stream_general:
                        general_response_content += part_general['message']['content']
                        print(part_general['message']['content'], end='', flush=True)
                    conversation_history.append({"role": "assistant", "content": general_response_content})

        except ollama.ResponseError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
                                                                                                           
def main():
    global faiss_index, transcript_chunks
    start_time = time.time()
    input_path = input("Enter YouTube URL, Video URL, or video file path: ").strip()
    first_question = input("\nEnter your question based on the video content: ").strip()

    audio_path = None
    if input_path.startswith("http"):
        audio_path = extract_audio_from_video_url(input_path)
    elif os.path.exists(input_path):
        audio_path = extract_audio(input_path)
    else:
        print("Error: Invalid input. Enter a valid URL or video file path.")
        return

    if not audio_path:
        return

    raw_transcript = transcribe_audio(audio_path)
    processed_transcript = preprocess_transcript(raw_transcript)
    punctuation_restored_transcript = restore_punctuation(processed_transcript)

    print("\n**Raw Transcript:**\n", raw_transcript)
    print("\n**Processed Transcript (Punctuation Restored):**\n", punctuation_restored_transcript)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    if processed_transcript:
        embed_and_build_faiss_index(processed_transcript) # Embed the processed transcript
        conversation_history = []
        relevant_context = None
        relevant_indices = None
        if faiss_index is not None and transcript_chunks:
            question_embedding = embedding_model.encode([first_question])[0].astype('float32').reshape(1, -1)
            k = 3
            distances, indices = faiss_index.search(question_embedding, k)
            if indices.any():
                relevant_context_chunks = [transcript_chunks[i] for i in indices[0]]
                relevant_context = "\n".join(relevant_context_chunks)
                relevant_indices = indices[0]

                print("\n**Relevant Transcript Chunks (Used for Initial Question):**")
                for i, chunk_index in enumerate(relevant_indices):
                    print(f"Chunk {chunk_index}:\n{transcript_chunks[chunk_index]}\n---")

        if relevant_context:
            initial_prompt = f"You are an AI assistant tasked with answering questions based on the following video transcript excerpts:\n\n{relevant_context}\n\nAnswer the following question based *only* on the provided excerpts. If the answer cannot be found in the excerpts, truthfully state that you cannot answer based on the provided information.\n\nQuestion: {first_question}"
        else:
            initial_prompt = f"Answer the following question based on the entire video transcript provided below. If the answer is not explicitly in the transcript, please indicate that. Transcript: {punctuation_restored_transcript}\n\nQuestion: {first_question}"

        conversation_history.append({"role": "user", "content": initial_prompt})

        try:
            stream = ollama.chat(
                model="llama3.2",
                messages=conversation_history,
                stream=True,
            )
            print("\n**Ollama Answer:**\n")
            response_content = ""
            for part in stream:
                response_content += part['message']['content']
                print(part['message']['content'], end='', flush=True)
            print("\n")
            conversation_history.append({"role": "assistant", "content": response_content})

            if "cannot answer based on the provided information" in response_content.lower() or "not explicitly in the transcript" in response_content.lower():
                pending_general_knowledge_question_main = first_question # Store the initial question
                confirm_general_knowledge = input("The answer wasn't found in the video content. Should I try to answer based on my general knowledge? (yes/no): ").strip().lower()
                if confirm_general_knowledge == "yes":
                    answer_question([{"role": "user", "content": f"Answer the following question based on your general knowledge: {pending_general_knowledge_question_main}"}])
                else:
                    answer_question(conversation_history)
            else:
                answer_question(conversation_history)

        except ollama.ResponseError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("Error: Could not process the transcript.")

    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()