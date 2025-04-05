document.addEventListener("DOMContentLoaded", function () {
    // ========================
    // DOM Element Selection
    // ========================
    const inputField = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const filePreviewContainer = document.getElementById("filePreviewContainer");
    const chatMessages = document.getElementById("chatMessages");
    const signinButton = document.getElementById("signin-button");
    const shareButton = document.querySelector(".share-btn");
    const inputContainer = document.querySelector(".input-container");
    let conversationHistory = []; // To store conversation history

    let selectedFiles = [];

    // ========================
    // Dynamic Text Area Expansion
    // ========================
    inputField.addEventListener("input", function () {
        this.style.height = "20px"; // Reset to single-line height
        this.style.height = Math.min(this.scrollHeight, 24 * 9) + "px"; // Expand but limit to 9 lines
    });

    // ========================
    // File Handling Functions
    // ========================

    /**
     * Updates the file preview section with selected files.
     */
    function updateFilePreview() {
        filePreviewContainer.innerHTML = "";
        inputContainer.style.minHeight = selectedFiles.length > 0 ? "160px" : "120px";
    
        selectedFiles.forEach(file => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");
    
            if (file.type.startsWith("image/")) {
                const img = document.createElement("img");
                img.src = URL.createObjectURL(file);
                img.style.maxWidth = "50px";
                fileTag.appendChild(img);
            }
    
            const fileName = document.createElement("span");
            fileName.textContent = file.name;
            fileTag.appendChild(fileName);
    
            filePreviewContainer.appendChild(fileTag);
        });
    }

    /**
     * Handles file selection when files are chosen.
     */
    fileInput.addEventListener("change", function () {
        selectedFiles = [...fileInput.files]; // Reset selected files to prevent duplication
        updateFilePreview();
    });

    // Opens the file manager when the attach button is clicked
    attachButton.addEventListener("click", function () {
        fileInput.click();
    });

    // ========================
    // Chat Message Functions
    // ========================

    /**
     * Sends a message and appends it to the chat area.
     */
    function sendMessage() {
        const messageText = inputField.value.trim();
        const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
        const youtubeUrlMatch = messageText.match(youtubeUrlRegex);

        if (youtubeUrlMatch) {
            const videoUrl = youtubeUrlMatch[0];
            const formData = new FormData();
            formData.append('videoUrl', videoUrl);
            const remainingText = messageText.replace(videoUrl, "").trim();
            inputField.value = remainingText; // Store the question in the input field for later
            processVideo(formData);
        } else if (selectedFiles.length > 0) {
            const formData = new FormData();
            formData.append('videoFile', selectedFiles[0]); // Assuming single file upload for now
            // If there's a question, store it for later
            const remainingText = messageText.trim();
            inputField.value = remainingText;
            processVideo(formData);
            selectedFiles = []; // Clear selected files after initiating upload
            updateFilePreview();
        } else if (messageText) {
            askQuestion(messageText);
            inputField.value = "";
        } else {
            displayMessage("Please enter a message, provide a Video URL, or attach a video file.", "system-message");
        }
    }

    sendButton.addEventListener("click", sendMessage);

    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // ... (rest of your other event listeners) ...

    fileInput.addEventListener("change", function () {
        selectedFiles = [...fileInput.files];
        updateFilePreview();
    });

    // ========================
    // Chat Sharing Function
    // ========================

    /**
     * Copies chat messages to the clipboard.
     */
    function shareChat() {
        let chatText = "";
        document.querySelectorAll(".user-message .message-text").forEach(msg => {
            chatText += msg.textContent + "\n\n";
        });

        if (!chatText.trim()) {
            alert("No chat messages to share.");
            return;
        }

        navigator.clipboard.writeText(chatText)
            .then(() => alert("Chat copied to clipboard!"))
            .catch(err => alert("Failed to copy chat: " + err));
    }

    function displayMessage(message, className) {
        const messageBubble = document.createElement("div");
        messageBubble.classList.add(className);
        const textNode = document.createElement("p");
        textNode.classList.add("message-text");
        textNode.textContent = message;
        messageBubble.appendChild(textNode);
        chatMessages.prepend(messageBubble);
    }

    function askQuestion(question) {
        displayMessage(`Question: ${question}`, "user-message");
        sendButton.disabled = true;
        sendButton.textContent = "Loading...";

        fetch('/api/ask_question/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }), // Only send the question
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                displayMessage(`Error: ${data.error}`, "error-message");
                return;
            }
            displayMessage(`Answer: ${data.answer}`, "assistant-message");
            // We don't need to manually update JavaScript's conversationHistory here
        })
        .catch(error => {
            displayMessage(`Network error: ${error.message}`, "error-message");
        })
        .finally(()=>{
            sendButton.disabled = false;
            sendButton.textContent = "Send";
        });
    }
    
    let processingVideo = false;

    function processVideo(videoData) {
        processingVideo = true;
        displayMessage("Processing video...", "system-message");
        sendButton.disabled = true;
        sendButton.textContent = "Loading...";

        fetch('/api/upload_video/', {
            method: 'POST',
            body: videoData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            processingVideo = false;
            if (data.error) {
                displayMessage(`Error: ${data.error}`, "error-message");
                return;
            }
            displayMessage("Audio extracted.", "system-message");
            displayMessage("Transcription completed.", "system-message");
            // The backend will handle storing the transcript in the session

            // NOW, after successful processing, if there's a question, ask it
            if (inputField.value.trim()) {
                askQuestion(inputField.value.trim());
                inputField.value = "";
            }
        })
        .catch(error => {
            processingVideo = false;
            displayMessage(`Network error: ${error.message}`, "error-message");
        })
        .finally(() => {
            sendButton.disabled = false;
            sendButton.textContent = "Send";
        });
    }

    // ========================
    // Event Listeners
    // ========================

    sendButton.addEventListener("click", function () {
        const messageText = inputField.value.trim();
        const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
        const youtubeUrlMatch = messageText.match(youtubeUrlRegex);
        if (youtubeUrlMatch) {
            const videoUrl = youtubeUrlMatch[0];
            const formData = new FormData();
            formData.append('videoUrl', videoUrl);
            processVideo(formData);
            inputField.value = messageText.replace(videoUrl, "").trim();
        } else if (messageText) {
            askQuestion(messageText);
            inputField.value = "";
        }
    });

    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    if (shareButton) {
        shareButton.addEventListener("click", shareChat);
    }

    signinButton.addEventListener("click", function () {
        alert("Sign-in button clicked! Implement authentication logic.");
    });

    attachButton.addEventListener("click", function () {
        fileInput.click();
    });

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('videoFile', file);
            processVideo(formData);
        }
        fileInput.value = "";
    });
});
