const viewTranscriptBtn = document.getElementById('viewTranscriptBtn');
const transcriptPopup = document.getElementById('transcriptPopup');
const transcriptContent = document.getElementById('transcriptContent');

// Forcefully hide the popup on initial load
if (transcriptPopup) {
    transcriptPopup.style.display = 'none';
    console.log("Transcript popup forcefully hidden on load.");
}

let isProcessing = false; // Flag to prevent multiple submissions
let currentVideoContext = false; // Flag to track if video context is loaded
let chatHistory = []; // Stores { role: 'user'/'assistant', content: '...' }

function showTranscript(transcriptText) {
    console.log("--- showTranscript function called ---"); // ADDED LOG
    console.log("showTranscript called with:", transcriptText);
    transcriptContent.textContent = transcriptText;
    transcriptPopup.style.display = 'block';
    viewTranscriptBtn.disabled = false;
    viewTranscriptBtn.style.opacity = 1;
    viewTranscriptBtn.style.cursor = 'pointer';
}

function closeTranscriptPopup() {
    transcriptPopup.style.display = 'none';
}

if (viewTranscriptBtn) {
    viewTranscriptBtn.addEventListener('click', function() {
        transcriptPopup.style.display = 'block';
    });
}

function focusInput() {
    document.getElementById("messageInput").focus();
}
window.onload = focusInput;
document.querySelector('.main-container').addEventListener('click', (event) => {
    if (event.target.tagName !== 'BUTTON' && event.target.tagName !== 'TEXTAREA' && event.target.id !== 'signin-button' && !event.target.closest('.sidebar')) {
        focusInput();
    }
});


function toggleLogoutButton() {
    var logoutButton = document.getElementById("logoutButton");
    logoutButton.style.display = logoutButton.style.display === "none" ? "block" : "none";
}

function confirmLogout() {
    if (confirm("Do you really want to logout?")) {
        window.location.href = "{{ logout_url }}";
    }
}

function setProcessingState(processing) {
    isProcessing = processing;
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const loadingIndicator = document.getElementById("loadingIndicator");

    if(messageInput) messageInput.disabled = processing;
    if(sendButton) sendButton.disabled = processing;
    if(attachButton) attachButton.disabled = processing;

    if (loadingIndicator) {
        loadingIndicator.style.display = processing ? "inline-block" : "none";
    }
    if (sendButton) {
        if (!processing) {
            sendButton.innerHTML = '<span class="material-icons">↑</span>';
        }
    }
}
setProcessingState(false);

function displayMessage(content, className, id = null) {
    const chatMessages = document.getElementById("chatMessages");
    if (!chatMessages) return null;

    const messageBubble = document.createElement("div");
    if (id) messageBubble.id = id;
    messageBubble.classList.add(className);

    const textContentElement = document.createElement("p");
    if (typeof content === 'string') {
        textContentElement.textContent = content;
    } else if (content instanceof HTMLElement) {
        textContentElement.appendChild(content);
    }
    messageBubble.appendChild(textContentElement);

    chatMessages.append(messageBubble); // Append to bottom

    const chatArea = document.querySelector(".chat-area");
    if (chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight; // Scroll down
    }
    return messageBubble;
}

function removeMessageById(id) {
    const messageToRemove = document.getElementById(id);
    if (messageToRemove) {
        messageToRemove.remove();
    }
}

function indexProcessVideo(videoData, initialQuestion = '') {
    console.log("--- indexProcessVideo function called ---"); // ADDED LOG
    setProcessingState(true);
    currentVideoContext = false; // Context is processing, not ready
    const processingStatusId = `status-${Date.now()}`;
    displayMessage("⏳ Uploading and processing video...", "system-message", processingStatusId);

    let fetchBody;
    let headers = { /* 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF if needed */ };

    if (videoData instanceof FormData) {
        fetchBody = videoData;
    } else {
        fetchBody = JSON.stringify(videoData);
        headers['Content-Type'] = 'application/json';
    }

    // API call to upload endpoint
    fetch('/api/upload_video/', { // Ensure URL matches urls.py
        method: 'POST',
        headers: headers,
        body: fetchBody,
    })
    .then(response => {
        console.log("--- indexProcessVideo: Response received ---", response); // ADDED LOG
        removeMessageById(processingStatusId);
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `Server error: ${response.status}`);
            }).catch(() => {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("--- indexProcessVideo: Data received ---", data); // ADDED LOG
        if (data.error) throw new Error(data.error);

        displayMessage(`✅ ${data.message || 'Video processed.'}`, "system-message");
        currentVideoContext = true; // Mark context as ready

        // ** Check if transcript is present and has content **
        if (data.transcript && data.transcript.length > 0) {
            console.log("Transcript found, calling showTranscript:", data.transcript); // ADDED LOG
            showTranscript(data.transcript);
        } else {
            console.log("No transcript in this response, enabling button anyway."); // ADDED LOG
            viewTranscriptBtn.disabled = false;
            viewTranscriptBtn.style.opacity = 1;
            viewTranscriptBtn.style.cursor = 'pointer';
        }

        // ** Initialize history if initial question was asked AND answered **
        if (initialQuestion && data.answer) {
            chatHistory = [
                { role: 'user', content: initialQuestion },
                { role: 'assistant', content: data.answer }
            ];
            console.log("Initialized history with initial Q&A.");
            // Display the initial answer properly
            const assistantBubble = displayMessage(data.answer, "assistant-message");
            // Optional Markdown/Sanitization
            const contentElement = assistantBubble?.querySelector('p');
            if (contentElement && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                try {
                    const rawHtml = marked.parse(data.answer);
                    contentElement.innerHTML = DOMPurify.sanitize(rawHtml);
                    if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
                } catch(e) { console.error("Markdown/DOMPurify error on initial answer:", e); }
            } else {
                // If no initial question or no answer came back, history remains empty
                chatHistory = [];
                console.log("No initial Q&A, history is empty.");
            }
        }
    })
    .catch(error => {
        console.error("--- indexProcessVideo: Processing Error ---", error);
        removeMessageById(processingStatusId);
        displayMessage(`❌ Error processing video: ${error.message}`, "error-message");
        currentVideoContext = false; // Context failed to load
        chatHistory = []; // Clear history on error too
    })
    .finally(() => {
        setProcessingState(false);
    });
}

const transcriptDataElement = document.getElementById('transcriptFromBackend');
if (transcriptDataElement && transcriptDataElement.textContent.length > 0) {
    showTranscript(transcriptDataElement.textContent);
}