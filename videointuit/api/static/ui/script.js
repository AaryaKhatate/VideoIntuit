/**
 * Frontend JavaScript for Chat Interface
 * Handles user input, file uploads, API calls, and streaming AI responses.
 * Includes client-side history management.
 */
document.addEventListener("DOMContentLoaded", function () {
    // ========================
    // DOM Element Selection
    // ========================
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const filePreviewContainer = document.getElementById("filePreviewContainer");
    const chatMessages = document.getElementById("chatMessages");
    const chatArea = document.querySelector(".chat-area");
    const inputContainer = document.querySelector(".input-container");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const shareButton = document.querySelector(".share-btn"); // Optional
    const analyzeFullTranscriptCheckbox = document.getElementById("analyzeFullTranscript"); // Example checkbox

    let selectedFiles = []; // Array to hold selected file objects for preview
    let isProcessing = false; // Flag to prevent multiple submissions
    let currentVideoContext = false; // Flag to track if video context is loaded

    // ** Client-side conversation history **
    let chatHistory = []; // Stores { role: 'user'/'assistant', content: '...' }

    // ========================
    // UI State Management
    // ========================
    function setProcessingState(processing) {
        isProcessing = processing;
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

    // ========================
    // Dynamic Text Area Expansion
    // ========================
    if (messageInput) {
        messageInput.addEventListener("input", function () {
            this.style.height = "auto";
            const scrollHeight = this.scrollHeight;
            const maxHeight = 150;
            this.style.height = Math.min(scrollHeight, maxHeight) + "px";
        });
    }

    // ========================
    // File Handling Functions
    // ========================
    // --- Keep updateFilePreview and removeFile functions exactly as they were ---
    function updateFilePreview() {
        if (!filePreviewContainer) return;

        filePreviewContainer.innerHTML = "";
        if(inputContainer) {
            inputContainer.style.paddingTop = selectedFiles.length > 0 ? "15px" : "12px";
        }

        selectedFiles.forEach((file, index) => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");

            let icon = '📄';
            if (file.type.startsWith("video/")) icon = '🎬';
            else if (file.type.startsWith("audio/")) icon = '🎵';
            else if (file.type.startsWith("image/")) icon = '🖼️';


            const iconSpan = document.createElement("span");
            iconSpan.textContent = icon;
            iconSpan.style.marginRight = "8px";
            fileTag.appendChild(iconSpan);

            const fileName = document.createElement("span");
            const maxLen = 25;
            fileName.textContent = file.name.length > maxLen
                ? file.name.substring(0, maxLen - 3) + '...'
                : file.name;
            fileTag.appendChild(fileName);

            const removeBtn = document.createElement("button");
            removeBtn.textContent = "✖";
            removeBtn.classList.add("remove-file-btn");
            removeBtn.style.background = 'none';
            removeBtn.style.border = 'none';
            removeBtn.style.color = '#888';
            removeBtn.style.marginLeft = '10px';
            removeBtn.style.cursor = 'pointer';
            removeBtn.style.fontSize = '14px';
            removeBtn.setAttribute('aria-label', `Remove ${file.name}`);

            removeBtn.onclick = (e) => {
                e.stopPropagation();
                removeFile(index);
            };
            fileTag.appendChild(removeBtn);
            filePreviewContainer.appendChild(fileTag);
        });
    }

    function removeFile(indexToRemove) {
        selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
        const dataTransfer = new DataTransfer();
        selectedFiles.forEach(file => dataTransfer.items.add(file));
        if (fileInput) {
            fileInput.files = dataTransfer.files;
        }
        updateFilePreview();
    }

    if (fileInput) {
        fileInput.addEventListener("change", function (event) {
            const newFiles = Array.from(event.target.files);
            // Clear existing selection preview before adding new ones
            selectedFiles = [];
            newFiles.forEach(newFile => {
                // Optional: Add checks for file type or size here if needed
                selectedFiles.push(newFile);
            });

            // Update the file input's internal list to match selection
            const dataTransfer = new DataTransfer();
            selectedFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;

            updateFilePreview();
        });
    }

    if (attachButton) {
        attachButton.addEventListener("click", function () {
            if (isProcessing || !fileInput) return;
            fileInput.click(); // Trigger hidden file input
        });
    }

    // ========================
    // Chat Message Functions
    // ========================
    function displayMessage(content, className, id = null) {
        if (!chatMessages) return null;

        const messageBubble = document.createElement("div");
        if (id) messageBubble.id = id;
        messageBubble.classList.add(className);

        const textContentElement = document.createElement("p");
        if (typeof content === 'string') {
             // Basic check for potential HTML, treat as text otherwise
             // More robust sanitization happens later if using Markdown
             textContentElement.textContent = content;
        } else if (content instanceof HTMLElement) {
             textContentElement.appendChild(content);
        }
        messageBubble.appendChild(textContentElement);

        chatMessages.append(messageBubble); // Append to bottom

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

    /**
     * Sends a text question (with history) to the backend and streams the response.
     * Updates client-side history on successful completion.
     * @param {string} question - The question text to send.
     */
    async function askQuestion(question) {
        setProcessingState(true);
        displayMessage(question, "user-message"); // Show user's question
         // ** Add user question to history immediately **
        chatHistory.push({ role: "user", content: question });

        const thinkingStatusId = `thinking-${Date.now()}`;
        displayMessage("🤔 Assistant is thinking...", "system-message", thinkingStatusId);

        let assistantBubble = null;
        let contentElement = null;
        let accumulatedAnswer = ""; // Store the full response text for history

        try {
            // API call, now includes client-side history
            const response = await fetch('/api/ask_question/', { // Ensure URL matches urls.py
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF if needed
                },
                // ** Send current question AND history **
                body: JSON.stringify({
                    question: question,
                    history: chatHistory.slice(0, -1) // Send history *before* this question
                }),
            });

            if (!response.ok) {
                removeMessageById(thinkingStatusId);
                let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                try {
                     const err = await response.json();
                     errorMsg = err.error || errorMsg;
                } catch { /* Ignore if error response isn't JSON */ }
                throw new Error(errorMsg);
            }

            // --- Response OK, prepare for streaming ---
            removeMessageById(thinkingStatusId);
            assistantBubble = displayMessage("", "assistant-message"); // Create empty bubble
            contentElement = assistantBubble?.querySelector('p');

            if (!contentElement) {
                throw new Error("Could not create assistant message bubble.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let reading = true;

            while (reading) {
                const { done, value } = await reader.read();
                if (done) {
                    reading = false;
                    break;
                }
                const chunk = decoder.decode(value, { stream: true });
                accumulatedAnswer += chunk; // Accumulate full response
                contentElement.textContent = accumulatedAnswer; // Update display

                if (chatArea) chatArea.scrollTop = chatArea.scrollHeight; // Keep scrolled down
            }

            // --- Stream finished ---
            // ** Add successful assistant response to history **
            if (accumulatedAnswer && !accumulatedAnswer.includes("--- Error:")) {
                 chatHistory.push({ role: "assistant", content: accumulatedAnswer });
            } else if (!accumulatedAnswer) {
                 // Handle case where stream finished but was empty (maybe backend yielded nothing)
                 chatHistory.push({ role: "assistant", content: "(AI returned no content)" });
            } // Don't add error messages to history as assistant turns

            // Optional: Apply Markdown/Sanitization to the final complete response
             if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                  try {
                       const rawHtml = marked.parse(accumulatedAnswer);
                       contentElement.innerHTML = DOMPurify.sanitize(rawHtml);
                       if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
                  } catch(e) { console.error("Markdown/DOMPurify error on final answer:", e); }
             }

        } catch (error) {
            console.error("Asking Question Error:", error);
            removeMessageById(thinkingStatusId); // Ensure thinking removed
            // Update the last user message in history to indicate failure? Or just display error?
            // Let's just display error for now.
             if (assistantBubble && contentElement) {
                 // If stream started, show error inline
                 contentElement.textContent = accumulatedAnswer + `\n\n--- Error: ${error.message} ---`;
                 // Remove the failed assistant turn from history if it was added prematurely
                 // (Current logic adds only after successful stream, which is safer)
             } else {
                 // If fetch failed before creating the bubble
                 displayMessage(`❌ Error: ${error.message}`, "error-message");
             }
             // Remove the user message that caused the error from history
             if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
                 chatHistory.pop();
             }
        } finally {
            setProcessingState(false); // Re-enable input fields
            console.log("Ask question processing finished. History length:", chatHistory.length);
        }
    }

    // ========================
    // Sending Logic (Main Action)
    // ========================
    function sendMessage() {
        if (isProcessing || !messageInput) return;

        let messageText = messageInput.value.trim();
        const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
        const youtubeUrlMatch = messageText.match(youtubeUrlRegex);
        let questionForVideo = "";

        // --- Determine Action Priority: File > URL > Text ---
        if (selectedFiles.length > 0) {
            // --- Action: Process File ---
             // ** A file upload implies a new context, clear history **
            console.log("New file upload detected, clearing client history.");
            chatHistory = [];
            currentVideoContext = false; // Mark context as not ready yet

            questionForVideo = messageText; // Assume any text is question for the file
            // ** IMPORTANT: Only process the FIRST selected file **
            const fileToProcess = selectedFiles[0];

            displayMessage(`Attaching file: ${fileToProcess.name}`, "user-message");
             if (questionForVideo) {
                 displayMessage(`Initial question: ${questionForVideo}`, "system-message");
                 // Add user's intent to history (even though history is cleared,
                 // this first entry helps if an initial answer comes back)
                 // Actually, let processVideo handle history init based on response.
             }

            const formData = new FormData();
            formData.append('videoFile', fileToProcess);
             if (questionForVideo) {
                 formData.append('question', questionForVideo);
             }

            processVideo(formData, questionForVideo); // Pass question separately for history init

            // Clear inputs *after* initiating the process
            selectedFiles = [];
            if(fileInput) fileInput.value = "";
            updateFilePreview();
            messageInput.value = "";
            messageInput.style.height = 'auto';

        } else if (messageText.startsWith("http://") || messageText.startsWith("https://")) {
            // --- Action: Process Any URL ---
            console.log("Generic URL detected, attempting to process.");
            chatHistory = [];
            currentVideoContext = false;

            questionForVideo = ""; // No specific question extraction from URL itself
            displayMessage(`Processing URL: ${messageText}`, "user-message");

            processVideo({ videoUrl: messageText, question: "" }, ""); // Send full URL
            messageInput.value = "";
            messageInput.style.height = 'auto';
        }else if (youtubeUrlMatch) {
            // --- Action: Process URL ---
            // ** A URL implies a new context, clear history **
            console.log("New URL detected, clearing client history.");
            chatHistory = [];
            currentVideoContext = false;

            const videoUrl = youtubeUrlMatch[0];
            questionForVideo = messageText.replace(videoUrl, "").trim();

            displayMessage(`Video URL: ${videoUrl}`, "user-message");
             if (questionForVideo) {
                 displayMessage(`Initial question: ${questionForVideo}`, "system-message");
             }

            // Send URL and optional question as JSON
            processVideo({ videoUrl: videoUrl, question: questionForVideo }, questionForVideo); // Pass question

            messageInput.value = "";
            messageInput.style.height = 'auto';

        } else if (messageText) {
            let questionToSend = messageText;
            if (analyzeFullTranscriptCheckbox && analyzeFullTranscriptCheckbox.checked) {
                questionToSend = "analyze full transcript " + messageText;
            }
            askQuestion(questionToSend);
            messageInput.value = "";
            messageInput.style.height = 'auto';
        } else {
            // --- Action: Nothing Entered ---
            displayMessage("Please enter a message, provide a Video URL, or attach a video file.", "system-message error-message");
        }
    }

    // Attach send logic to button and Enter key
    if (sendButton) sendButton.addEventListener("click", sendMessage);
    if (messageInput) {
        messageInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }

    // ========================
    // Backend Interaction Functions
    // ========================

    /**
     * Sends video file or URL (with optional initial question) to the backend.
     * Initializes client-side history if an initial answer is received.
     * @param {FormData|Object} videoData - Data to send (FormData or {videoUrl, question}).
     * @param {string} [initialQuestion=''] - The initial question asked, for history init.
     */
    function processVideo(videoData, initialQuestion = '') {
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
            if (data.error) throw new Error(data.error);

            displayMessage(`✅ ${data.message || 'Video processed.'}`, "system-message");
            currentVideoContext = true; // Mark context as ready

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
                  }
            } else {
                // If no initial question or no answer came back, history remains empty
                chatHistory = [];
                console.log("No initial Q&A, history is empty.");
            }
        })
        .catch(error => {
            removeMessageById(processingStatusId);
            console.error("Processing Error:", error);
            displayMessage(`❌ Error processing video: ${error.message}`, "error-message");
            currentVideoContext = false; // Context failed to load
            chatHistory = []; // Clear history on error too
        })
        .finally(() => {
            setProcessingState(false);
        });
    }


    // ========================
    // Chat Sharing Function (Uses client history)
    // ========================
    function shareChat() {
        if (chatHistory.length === 0) {
             alert("No chat history to share.");
             return;
        }

        let chatText = "Chat History:\n\n";
        chatHistory.forEach(msg => {
             // Simple formatting, adjust as needed
             chatText += `${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}: ${msg.content}\n\n`;
        });

        navigator.clipboard.writeText(chatText.trim())
            .then(() => alert("Chat history copied to clipboard!"))
            .catch(err => {
                console.error("Failed to copy chat:", err);
                alert("Failed to copy chat. See console for details.");
            });
    }
     if (shareButton) {
         shareButton.addEventListener('click', shareChat);
     }

    // ========================
    // CSRF Token Helper (If needed)
    // ========================
    /* function getCookie(name) { ... } */

}); // End DOMContentLoaded