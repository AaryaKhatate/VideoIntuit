# VideoIntuit 🎬

> An AI-powered video analysis and Q&A system that allows users to upload videos or provide YouTube URLs, automatically transcribe them, and have intelligent conversations about the content using RAG (Retrieval Augmented Generation) and Large Language Models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Django](https://img.shields.io/badge/Django-5.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [System Design](#system-design)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Security Features](#security-features)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**VideoIntuit** is an intelligent video content analysis platform that combines cutting-edge AI technologies to enable natural language interactions with video content. Upload any video file or YouTube URL, and ask questions about its content - the system will provide accurate, context-aware answers by analyzing the video's transcript and optionally finding related YouTube videos for additional context.

### What Makes VideoIntuit Special?

- **Multi-Source Intelligence**: Analyzes both your uploaded video and optionally searches for related YouTube videos to provide comprehensive answers
- **Local AI Processing**: Uses Ollama for privacy-focused, on-device LLM processing
- **Advanced RAG**: Implements Retrieval Augmented Generation with FAISS vector search for precise context retrieval
- **Real-time Streaming**: Get answers as they're generated with streaming responses
- **Automatic Transcription**: Uses Whisper AI for accurate speech-to-text conversion
- **Session Management**: Maintains conversation history for contextual follow-up questions

---

## ✨ Key Features

### 🎥 Video Processing
- **Multiple Input Sources**: Upload local video files or provide YouTube URLs
- **Automatic Transcription**: Powered by OpenAI's Whisper model with GPU acceleration support
- **Audio Extraction**: FFmpeg-based audio extraction from various video formats
- **Format Support**: Handles MP4, AVI, MOV, and other common video formats

### 🤖 AI-Powered Q&A
- **Context-Aware Answers**: Uses the video transcript as primary context
- **RAG Implementation**: Semantic search over transcript chunks using sentence embeddings
- **Related Content Discovery**: Optionally finds and analyzes related YouTube videos
- **Streaming Responses**: Real-time answer generation with visual feedback
- **Conversation History**: Maintains context across multiple questions

### 🔍 NLP
- **Semantic Embeddings**: Uses Sentence Transformers for text understanding
- **Vector Search**: FAISS-powered similarity search for relevant context retrieval
- **Text Processing**: spaCy integration for advanced text analysis
- **Spell Correction**: Automatic correction of transcription errors

### 🔒 Security & Privacy
- **Environment-Based Configuration**: Secrets managed via .env files
- **CSRF Protection**: Full Cross-Site Request Forgery protection
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Session Security**: Secure cookie handling and session management
- **Local Processing**: AI models run locally via Ollama (no data sent to external APIs)

---

## 🏗️ Architecture

VideoIntuit follows a modern, modular architecture designed for scalability and maintainability.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Browser)                      │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────┐     │
│  │ Video Upload │  │ Chat Interface│  │ History Management│     │
│  └──────────────┘  └───────────────┘  └───────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Django Backend (API Layer)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Views (views.py)                  │   │
│  │  • upload_video()  • ask_question()  • index()           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│  ┌──────────────┬──────────┴────────┬──────────────────────┐    │
│  │              │                   │                      │    │
│  ▼              ▼                   ▼                      ▼    │
│ ┌────────┐  ┌────────┐         ┌─────────┐           ┌────────┐ │
│ │Session │  │ Cache  │         │ YouTube │           │ Models │ │
│ │Manager │  │ Layer  │         │   API   │           │ (DB)   │ │
│ └────────┘  └────────┘         └─────────┘           └────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI Processing Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Whisper    │  │   Ollama     │  │  Sentence Transformers│  │
│  │ (Transcribe) │  │   (LLM)      │  │   (Embeddings)        │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    spaCy     │  │    FAISS     │  │      FFmpeg          │   │
│  │    (NLP)     │  │ (Vector DB)  │  │  (Audio Extract)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### **Frontend Layer**
- **Technology**: Vanilla JavaScript with modern ES6+ features
- **UI Components**: Custom chat interface with file upload, message history, and streaming support
- **State Management**: Client-side conversation history management
- **Communication**: Fetch API with CSRF token handling for secure requests

#### **API Layer (Django)**
- **Framework**: Django 5.1 with MVT (Model-View-Template) pattern
- **Views**: Function-based views for video processing and Q&A
- **Middleware**: CORS, CSRF, Session, Security headers
- **Cache**: Django cache framework (configurable: in-memory, Redis, Memcached)

#### **AI Processing Layer**
1. **Whisper (OpenAI)**: Automatic speech recognition for video transcription
2. **Ollama**: Local LLM server for chat completions (Llama 3.2)
3. **Sentence Transformers**: Generate embeddings for semantic search
4. **FAISS**: Vector similarity search for RAG implementation
5. **spaCy**: NLP processing for text understanding
6. **FFmpeg**: Audio/video processing and conversion

---

## 🛠️ Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Core programming language |
| Django | 5.1 | Web framework |
| python-dotenv | 1.0+ | Environment variable management |
| django-cors-headers | 4.0+ | CORS handling |

### AI & Machine Learning
| Technology | Version | Purpose |
|------------|---------|---------|
| Ollama | Latest | Local LLM inference (Llama 3.2) |
| faster-whisper | Latest | Speech-to-text transcription |
| sentence-transformers | 2.0+ | Text embeddings |
| torch | 2.0+ | Deep learning framework |
| faiss-cpu/gpu | Latest | Vector similarity search |
| spaCy | 3.0+ | NLP processing |

### Video Processing
| Technology | Version | Purpose |
|------------|---------|---------|
| yt-dlp | 2023+ | YouTube video downloading |
| youtube-transcript-api | 0.6+ | YouTube transcript fetching |
| FFmpeg | Latest | Audio extraction |

### NLP & Utilities
| Technology | Version | Purpose |
|------------|---------|---------|
| NLTK | 3.6+ | Natural language toolkit |
| pyspellchecker | 0.7+ | Spell checking |
| NumPy | Latest | Numerical computing |
| Requests | Latest | HTTP client |

### Frontend
| Technology | Purpose |
|------------|---------|
| HTML5 | Structure |
| CSS3 | Styling |
| JavaScript (ES6+) | Interactivity |
| Material Icons | UI icons |

---

## 🎨 System Design

### Data Flow

```
┌─────────────┐
│ User Uploads│
│  Video/URL  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│ 1. Video Processing             │
│    • Save/Download video        │
│    • Extract audio (FFmpeg)     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 2. Transcription                │
│    • Whisper Model              │
│    • Generate raw transcript    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 3. Text Processing              │
│    • Spell correction           │
│    • NLP processing (spaCy)     │
│    • Clean & normalize          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 4. Optional: Find Related Videos│
│    • YouTube API search         │
│    • Fetch transcripts          │
│    • Process & combine          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 5. RAG Setup                    │
│    • Chunk transcripts          │
│    • Generate embeddings        │
│    • Store in cache + FAISS     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 6. User Asks Question           │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 7. Semantic Search              │
│    • Embed question             │
│    • FAISS vector search        │
│    • Retrieve relevant chunks   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ 8. LLM Generation               │
│    • Build prompt with context  │
│    • Call Ollama (streaming)    │
│    • Return answer to user      │
└─────────────────────────────────┘
```

### RAG (Retrieval Augmented Generation) Implementation

```
Question: "What is the main topic discussed?"
    ↓
[Embedding Model]
    ↓
Question Vector: [0.23, -0.45, 0.67, ...]
    ↓
[FAISS Vector Search]
    ↓
Top-K Similar Chunks:
  1. Chunk 45: "The main topic is AI..."
  2. Chunk 12: "We discuss machine learning..."
  3. Chunk 78: "Artificial intelligence has..."
    ↓
[Combine with System Prompt]
    ↓
Ollama Input:
  System: "Answer based on this context: [chunks]"
  User: "What is the main topic discussed?"
    ↓
[Ollama LLM - Streaming]
    ↓
Answer: "The main topic discussed is artificial 
         intelligence and machine learning..."
```

### Caching Strategy

- **Session-Based Caching**: Each user session has isolated cache
- **Cached Data**:
  - Full video transcript
  - YouTube-related video chunks
  - Pre-computed embeddings
- **Cache Keys**: `{data_type}_{session_key}`
- **TTL**: Configurable (default: 1 hour)

---

## 🚀 Setup & Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **FFmpeg** ([Download](https://ffmpeg.org/download.html))
- **Ollama** ([Download](https://ollama.ai/))
- **Git** ([Download](https://git-scm.com/downloads))

### Step 1: Clone the Repository

```powershell
git clone https://github.com/AaryaKhatate/Library-Management-System-GateGaurd-.git
cd Library-Management-System-GateGaurd-/videointuit
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download NLP Models

```powershell
# Download spaCy English model
python -m spacy download en_core_web_sm

# NLTK punkt tokenizer will download automatically on first run
```

### Step 5: Install Ollama

```powershell
# Install Ollama using winget
winget install Ollama.Ollama

# Or download from https://ollama.ai/
```

### Step 6: Pull Ollama Model

```powershell
# Pull the Llama 3.2 model (3B parameters, ~2GB)
ollama pull llama3.2

# Verify installation
ollama list
```

### Step 7: Install FFmpeg

**Option A: Using winget**
```powershell
winget install FFmpeg
```

**Option B: Manual Installation**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a directory (e.g., `C:\ffmpeg`)
3. Add to PATH: `C:\ffmpeg\bin`

**Verify installation:**
```powershell
ffmpeg -version
```

### Step 8: Configure Environment Variables

1. **Copy the example environment file:**
   ```powershell
   cp .env.example .env
   ```

2. **Generate a Django secret key:**
   ```powershell
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

3. **Edit `.env` file:**
   ```env
   # Required Settings
   DJANGO_SECRET_KEY=your-generated-secret-key-here
   DJANGO_DEBUG=True
   DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
   
   # YouTube API (Optional - for related video search)
   YOUTUBE_API_KEY=your-youtube-api-key-here
   
   # CORS (Optional - leave empty for development)
   CORS_ALLOWED_ORIGINS=
   
   # AI Models (Defaults are fine)
   OLLAMA_MODEL=llama3.2
   WHISPER_MODEL_NAME=medium
   ```

4. **Get YouTube API Key (Optional):**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a project
   - Enable YouTube Data API v3
   - Create credentials (API Key)
   - Add to `.env`

### Step 9: Database Setup

```powershell
# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### Step 10: Start Ollama Service

```powershell
# Ollama should start automatically on Windows
# If not, start it manually:
ollama serve
```

Keep this terminal open or run Ollama as a service.

### Step 11: Run the Development Server

```powershell
# In a new terminal (with venv activated)
python manage.py runserver
```

### Step 12: Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

---

## ⚙️ Configuration

### Environment Variables

All configuration is managed through `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | Yes | - | Django secret key (generate new one) |
| `DJANGO_DEBUG` | Yes | False | Debug mode (True for dev, False for prod) |
| `DJANGO_ALLOWED_HOSTS` | Yes | localhost | Comma-separated allowed hosts |
| `YOUTUBE_API_KEY` | No | - | YouTube Data API v3 key |
| `CORS_ALLOWED_ORIGINS` | No | - | Comma-separated CORS origins |
| `OLLAMA_MODEL` | No | llama3.2 | Ollama model to use |
| `WHISPER_MODEL_NAME` | No | medium | Whisper model size |
| `SPACY_MODEL_NAME` | No | en_core_web_sm | spaCy model |
| `EMBEDDING_MODEL_NAME` | No | all-MiniLM-L6-v2 | Sentence transformer model |
| `FFMPEG_COMMAND` | No | ffmpeg | FFmpeg executable path |
| `CHAT_CACHE_TIMEOUT` | No | 3600 | Cache timeout in seconds |
| `TRANSCRIPT_CHUNK_SIZE` | No | 300 | Characters per chunk for RAG |
| `RAG_TOP_K` | No | 4 | Number of chunks to retrieve |
| `MAX_HISTORY_TURNS` | No | 10 | Max conversation history |

### Model Selection

#### Whisper Models
Choose based on accuracy vs speed tradeoff:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75MB | Fastest | Basic |
| base | ~150MB | Fast | Good |
| small | ~500MB | Medium | Better |
| **medium** | ~1.5GB | **Recommended** | **Excellent** |
| large | ~3GB | Slow | Best |

#### Ollama Models
Alternative models you can use:

```powershell
# Smaller, faster model (1B parameters)
ollama pull llama3.2:1b

# Larger, more capable model (8B parameters)
ollama pull llama3.1:8b

# Best quality (70B parameters - requires powerful GPU)
ollama pull llama3.1:70b
```

Update `.env` to switch models:
```env
OLLAMA_MODEL=llama3.1:8b
```

---

## 💻 Usage

### Upload a Video

1. **Navigate to the home page**
2. **Click the attach button (📎)** or drag-and-drop a video file
3. **Supported formats**: MP4, AVI, MOV, MKV, WEBM
4. **Optional**: Check "Find related YouTube videos" for enhanced context
5. **Optional**: Enter an initial question
6. **Click Send** to start processing

### Provide a YouTube URL

1. **Enter a YouTube URL** in the message input
2. **Format**: `https://www.youtube.com/watch?v=VIDEO_ID`
3. **Optional**: Check "Find related YouTube videos"
4. **Click Send**

### Ask Questions

After video processing completes:

1. **Type your question** in the input field
2. **Questions can be about**:
   - Content summary
   - Specific topics mentioned
   - Explanations of concepts
   - Timeline of events
   - Key takeaways
3. **Receive streaming answers** in real-time
4. **Follow-up questions** maintain conversation context

### Advanced Features

#### Related Video Search
- Enable "Find related YouTube videos" checkbox
- System searches YouTube for related content
- Combines multiple transcripts for comprehensive answers
- Useful for educational content and tutorials

#### General Knowledge Mode
- If the system can't answer from the video alone
- It will ask permission to use general knowledge
- Type "yes" to allow or "no" to decline

---

## 📁 Project Structure

```
videointuit/
├── .env                          # Environment variables (DO NOT COMMIT)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── manage.py                     # Django management script
├── db.sqlite3                    # SQLite database
│
├── videointuit/                  # Main Django project
│   ├── __init__.py
│   ├── settings.py               # Django settings (uses .env)
│   ├── urls.py                   # Main URL routing
│   ├── views.py                  # Main page view
│   ├── wsgi.py                   # WSGI config
│   └── asgi.py                   # ASGI config
│
├── api/                          # API application
│   ├── __init__.py
│   ├── views.py                  # API endpoints (main logic)
│   ├── urls.py                   # API URL routing
│   ├── models.py                 # Database models
│   ├── admin.py                  # Django admin config
│   ├── apps.py                   # App configuration
│   │
│   ├── templates/                # HTML templates
│   │   ├── base.html             # Base template
│   │   └── index.html            # Main chat interface
│   │
│   ├── static/                   # Static files
│   │   ├── ui/
│   │   │   └── script.js         # Frontend JavaScript
│   │   └── images/               # Image assets
│   │
│   └── migrations/               # Database migrations
│
├── users/                        # User authentication app
│   ├── __init__.py
│   ├── views.py                  # Auth views
│   ├── urls.py                   # Auth URL routing
│   ├── forms.py                  # Authentication forms
│   ├── models.py                 # User models
│   │
│   └── templates/                # Auth templates
│       ├── signup.html
│       └── login.html
│
├── staticfiles/                  # Collected static files (production)
└── config.py                     # Legacy config (deprecated)
```

### Key Files Explained

#### `api/views.py` (Core Logic - 1462 lines)
- **Video Processing Functions**:
  - `upload_video()`: Handles video upload and transcription
  - `extract_audio_from_file()`: FFmpeg audio extraction
  - `extract_audio_from_video_url()`: yt-dlp download & extraction
  - `transcribe_audio()`: Whisper transcription
  
- **Q&A Functions**:
  - `ask_question()`: Handles user questions with RAG
  - `get_ollama_response_stream()`: Streaming LLM responses
  - `call_llm()`: Non-streaming LLM calls
  
- **NLP Functions**:
  - `preprocess_transcript()`: Text cleaning and normalization
  - `chunk_text()`: Split text into chunks for RAG
  - `extract_keywords()`: Keyword extraction for search
  
- **YouTube Integration**:
  - `search_youtube()`: YouTube API search
  - `get_youtube_transcript()`: Fetch YouTube transcripts
  - `find_related_youtube_videos()`: Multi-video analysis

#### `static/ui/script.js` (Frontend - 683 lines)
- CSRF token handling
- File upload with preview
- Message display system
- Streaming response handling
- Conversation history management
- Error handling and UI feedback

#### `settings.py`
- Environment variable loading
- Security configurations
- CORS and CSRF settings
- Database configuration
- Static files handling

---

## 🔌 API Endpoints

### POST `/api/upload_video/`

Upload a video file or URL for processing.

**Request (File Upload):**
```javascript
FormData {
  videoFile: File,
  question: String (optional),
  find_other_videos: Boolean (optional)
}
```

**Request (URL):**
```json
{
  "videoUrl": "https://youtube.com/watch?v=...",
  "question": "What is this video about?",
  "find_other_videos": false
}
```

**Response:**
```json
{
  "message": "Video processed successfully",
  "transcript_preview": "First 500 characters...",
  "initial_answer": "Answer to initial question...",
  "related_videos": [
    {"title": "Video 1", "url": "..."},
    {"title": "Video 2", "url": "..."}
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid input)
- `503`: Service unavailable (models not loaded)
- `500`: Server error

---

### POST `/api/ask_question/`

Ask a question about the uploaded video.

**Request:**
```json
{
  "question": "What is the main topic?",
  "history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ],
  "find_other_videos": false
}
```

**Response:**
Streaming text/plain response with answer chunks.

**Status Codes:**
- `200`: Success (streaming)
- `400`: Bad request / session expired
- `503`: Service unavailable
- `500`: Server error

---

## 🔒 Security Features

### Implemented Security Measures

1. **Environment-Based Secrets**
   - No hardcoded API keys or secrets
   - `.env` file for all sensitive data
   - `.gitignore` prevents accidental commits

2. **CSRF Protection**
   - Django CSRF middleware enabled
   - CSRF tokens in all forms and AJAX requests
   - Cookie-based token validation

3. **CORS Configuration**
   - Configurable allowed origins
   - Restrictive by default in production
   - Only allows specified domains

4. **Secure Headers**
   - `X-Frame-Options: DENY`
   - `X-Content-Type-Options: nosniff`
   - `X-XSS-Protection: 1; mode=block`
   - HTTPS redirect in production

5. **Session Security**
   - HTTP-only cookies
   - Secure cookies in production
   - SameSite cookie attribute
   - Session timeout

6. **Input Validation**
   - File type validation
   - URL validation
   - Content length limits
   - SQL injection prevention (Django ORM)

7. **Error Handling**
   - Generic error messages to users
   - Detailed logging for developers
   - No stack traces in production

### Production Deployment Checklist

Before deploying to production:

- [ ] Set `DJANGO_DEBUG=False`
- [ ] Generate new `DJANGO_SECRET_KEY`
- [ ] Configure `DJANGO_ALLOWED_HOSTS`
- [ ] Set up `CORS_ALLOWED_ORIGINS`
- [ ] Enable HTTPS/SSL
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up proper logging
- [ ] Configure Redis for caching
- [ ] Set up monitoring and alerts
- [ ] Regular security updates
- [ ] Rate limiting on API endpoints
- [ ] Implement file size limits

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```
Error: Could not connect to AI model service
```

**Solution:**
```powershell
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Verify model is installed
ollama pull llama3.2
```

---

#### 2. FFmpeg Not Found
```
FFmpeg command not found
```

**Solution:**
```powershell
# Install FFmpeg
winget install FFmpeg

# Or add to PATH manually
# Control Panel → System → Advanced → Environment Variables
# Add FFmpeg bin directory to PATH

# Verify
ffmpeg -version
```

---

#### 3. Whisper Model Loading Error
```
CRITICAL: Failed to load Whisper model
```

**Solutions:**
- Check available disk space (models are large)
- Verify internet connection for first download
- Try a smaller model: `.env` → `WHISPER_MODEL_NAME=small`
- For GPU issues: Install CUDA toolkit or use CPU mode

---

#### 4. YouTube API Quota Exceeded
```
YouTube API quota exceeded
```

**Solutions:**
- Wait 24 hours for quota reset
- Get additional quota from Google Cloud Console
- Disable "Find related videos" feature temporarily
- Use different API key

---

#### 5. Session Expired Error
```
Error: Your session has expired
```

**Solution:**
- Refresh the page
- Upload the video again
- Check cache timeout in `.env`: `CHAT_CACHE_TIMEOUT=7200`

---

#### 6. CSRF Token Missing
```
CSRF verification failed
```

**Solution:**
- Clear browser cookies
- Refresh the page
- Check browser console for JavaScript errors
- Verify CSRF middleware is enabled in settings

---

### Debug Mode

Enable detailed logging:

```env
DJANGO_DEBUG=True
```

Check logs in terminal where Django is running.

---

## 🚀 Future Improvements

### Short-term Enhancements

#### 1. **Rate Limiting**
- Implement per-IP request limiting
- Prevent API abuse
- Use `django-ratelimit` package

```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='10/m')
def upload_video(request):
    ...
```

#### 2. **File Size Limits**
- Add configurable upload size limits
- Implement chunked uploads for large files
- Progress tracking for uploads

#### 3. **Better Error Messages**
- User-friendly error explanations
- Suggested actions for common errors
- Multi-language support

#### 4. **Video Timestamps**
- Link answers to specific video timestamps
- "Jump to section" functionality
- Visual timeline scrubber

#### 5. **Export Functionality**
- Export conversation as PDF/Markdown
- Save transcripts as text files
- Share conversation via link

---

### Medium-term Features

#### 6. **Multi-Modal Analysis**
- Analyze video frames (computer vision)
- Detect objects and scenes
- OCR for text in videos
- Face recognition

#### 7. **Real-time Collaboration**
- Multiple users can analyze same video
- Shared conversation sessions
- WebSocket for real-time updates

#### 8. **Advanced Search**
- Full-text search across all videos
- Filter by date, length, topic
- Semantic search across library

#### 9. **Playlist Support**
- Upload multiple videos
- Analyze entire playlists
- Cross-video questions

#### 10. **Mobile App**
- React Native or Flutter app
- Mobile-optimized UI
- Offline mode support

---

### Long-term Vision

#### 11. **Enterprise Features**
- Team workspaces
- Role-based access control
- Admin dashboard
- Usage analytics

#### 12. **Advanced AI**
- Custom fine-tuned models
- Domain-specific models (medical, legal, etc.)
- Multi-language support
- Sentiment analysis

#### 13. **Integrations**
- Zoom meeting transcription
- Microsoft Teams integration
- Slack bot
- Google Drive sync
- Dropbox integration

#### 14. **Performance Optimizations**
- GPU optimization
- Distributed processing
- CDN for static files
- Database query optimization
- Caching strategies

#### 15. **Scalability**
- Kubernetes deployment
- Horizontal scaling
- Load balancing
- Microservices architecture
- Message queue (Celery/RabbitMQ)

---

### Research & Experimental

#### 16. **Advanced RAG**
- Hierarchical document structure
- Graph-based RAG
- Multi-hop reasoning
- Citations and sources

#### 17. **Multimodal RAG**
- Combine text, audio, and visual features
- Cross-modal retrieval
- Video understanding models

#### 18. **Custom Training**
- Fine-tune models on domain data
- Continuous learning from user feedback
- Active learning strategies

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint for JavaScript
- Write descriptive commit messages
- Add tests for new features
- Update documentation
- Comment complex logic

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🌐 Translations
- 🧪 Testing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** - Whisper speech recognition model
- **Meta** - Llama language models
- **Sentence Transformers** - Text embedding models
- **Facebook AI Research** - FAISS vector search
- **spaCy** - Industrial-strength NLP
- **Django** - Web framework
- **Ollama** - Local LLM runtime
- **FFmpeg** - Multimedia processing

---



---

<div align="center">

**Made with ❤️ by the VideoIntuit Team**

⭐ Star this repo if you find it helpful!

</div>
