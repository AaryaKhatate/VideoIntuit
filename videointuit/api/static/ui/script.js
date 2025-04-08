/**
 * Frontend JavaScript for Chat Interface
 * Handles user input, file uploads, API calls, and streaming AI responses.
 */
document.addEventListener("DOMContentLoaded", function () {
    // ========================
    // DOM Element Selection
    // ========================
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn"); // Ensure ID matches HTML
    const attachButton = document.getElementById("attachBtn"); // Ensure ID matches HTML
    const fileInput = document.getElementById("fileInput"); // Ensure ID matches HTML (hidden)
    const filePreviewContainer = document.getElementById("filePreviewContainer"); // Ensure ID matches HTML
    const chatMessages = document.getElementById("chatMessages"); // Container for message bubbles
    const chatArea = document.querySelector(".chat-area"); // The scrollable container itself
    const signinButton = document.getElementById("signin-button"); // Optional
    const shareButton = document.querySelector(".share-btn"); // Optional
    const inputContainer = document.querySelector(".input-container");
    // Ensure you have <div id="loadingIndicator" style="display: none;"></div> (e.g., a spinner)
    const loadingIndicator = document.getElementById("loadingIndicator");

    let selectedFiles = []; // Array to hold selected file objects
    let isProcessing = false; // Flag to prevent multiple submissions

    // ========================
    // UI State Management
    // ========================
    function setProcessingState(processing) {
        isProcessing = processing;
        // Disable/enable inputs during processing
        if(messageInput) messageInput.disabled = processing;
        if(sendButton) sendButton.disabled = processing;
        if(attachButton) attachButton.disabled = processing;

        // Show/hide loading indicator (e.g., spinner)
        if (loadingIndicator) {
            loadingIndicator.style.display = processing ? "inline-block" : "none";
        }

        // Optional: Change send button icon/appearance during processing
        if (sendButton) {
            if (!processing) {
                 // Restore default send icon (assuming Material Icons, adjust if not)
                 sendButton.innerHTML = '<span class="material-icons">↑</span>';
            } else {
                 // Optional: Show a processing indicator on the button itself
                 // sendButton.innerHTML = '<span class="material-icons">hourglass_empty</span>';
            }
        }
    }
    // Set initial state when the page loads
    setProcessingState(false);

    // ========================
    // Dynamic Text Area Expansion
    // ========================
    if (messageInput) {
        messageInput.addEventListener("input", function () {
            this.style.height = "auto"; // Reset height
            const scrollHeight = this.scrollHeight;
            const maxHeight = 150; // Define a max-height (pixels)
            // Set height based on content, up to max height
            this.style.height = Math.min(scrollHeight, maxHeight) + "px";
            // CSS 'overflow: auto;' should handle scrolling beyond max-height
        });
    }

    // ========================
    // File Handling Functions
    // ========================
    function updateFilePreview() {
        if (!filePreviewContainer) return; // Exit if container doesn't exist

        filePreviewContainer.innerHTML = ""; // Clear existing previews
        // Adjust container padding slightly based on whether files are present
        if(inputContainer) {
            inputContainer.style.paddingTop = selectedFiles.length > 0 ? "15px" : "12px";
        }

        selectedFiles.forEach((file, index) => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag"); // Use class from your CSS

            // Determine a simple icon based on file type
            let icon = '📄'; // Default file icon
            if (file.type.startsWith("image/")) icon = '🖼️'; // Image icon
            if (file.type.startsWith("video/")) icon = '🎬'; // Video icon
            if (file.type.startsWith("audio/")) icon = '🎵'; // Audio icon

            const iconSpan = document.createElement("span");
            iconSpan.textContent = icon;
            iconSpan.style.marginRight = "8px"; // Add some space after the icon
            fileTag.appendChild(iconSpan);

            // Display filename, truncated if too long
            const fileName = document.createElement("span");
            const maxLen = 25;
            fileName.textContent = file.name.length > maxLen
                ? file.name.substring(0, maxLen - 3) + '...'
                : file.name;
            fileTag.appendChild(fileName);

            // Add a remove button ('✖') for this file
            const removeBtn = document.createElement("button");
            removeBtn.textContent = "✖";
            removeBtn.classList.add("remove-file-btn"); // Add class for specific styling
            // Basic inline styles for the remove button (better to style via CSS class)
            removeBtn.style.background = 'none';
            removeBtn.style.border = 'none';
            removeBtn.style.color = '#888'; // Adjust color as needed
            removeBtn.style.marginLeft = '10px';
            removeBtn.style.cursor = 'pointer';
            removeBtn.style.fontSize = '14px';
            removeBtn.setAttribute('aria-label', `Remove ${file.name}`); // Accessibility

            // Add click handler to remove the file
            removeBtn.onclick = (e) => {
                e.stopPropagation(); // Prevent clicks triggering other actions
                removeFile(index);
            };
            fileTag.appendChild(removeBtn);

            // Add the completed tag to the preview container
            filePreviewContainer.appendChild(fileTag);
        });
    }

    function removeFile(indexToRemove) {
        // Filter out the file at the specified index
        selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);

        // Update the actual <input type="file"> element's file list
        const dataTransfer = new DataTransfer();
        selectedFiles.forEach(file => dataTransfer.items.add(file));
        if (fileInput) {
            fileInput.files = dataTransfer.files;
        }

        updateFilePreview(); // Refresh the visual preview area
    }

    if (fileInput) {
        fileInput.addEventListener("change", function (event) {
            const newFiles = Array.from(event.target.files);

            // Add newly selected files, avoiding duplicates (based on name & size)
            newFiles.forEach(newFile => {
                if (!selectedFiles.some(existingFile =>
                    existingFile.name === newFile.name && existingFile.size === newFile.size
                )) {
                    selectedFiles.push(newFile);
                }
            });

            // Update the file input's internal list to reflect the potentially filtered list
            const dataTransfer = new DataTransfer();
            selectedFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;

            updateFilePreview(); // Update the visual preview
        });
    }

    if (attachButton) {
        attachButton.addEventListener("click", function () {
            if (isProcessing || !fileInput) return; // Don't allow attaching while busy
            fileInput.click();
        });
    }

    // ========================
    // Chat Message Functions
    // ========================

    /**
     * Creates and displays a message bubble in the chat area.
     * Uses append() for standard chat order (newest at bottom).
     * @param {string | HTMLElement} content - The text or HTML content for the message.
     * @param {string} className - The CSS class for the bubble (e.g., 'user-message').
     * @param {string} [id] - Optional unique ID for the message bubble element.
     * @returns {HTMLElement} The created message bubble element.
     */
    function displayMessage(content, className, id = null) {
        if (!chatMessages) return null; // Exit if chat container doesn't exist

        const messageBubble = document.createElement("div");
        if (id) {
            messageBubble.id = id; // Assign ID if provided
        }
        messageBubble.classList.add(className); // Apply styling class

        // Use a <p> tag internally for better structure and targeting
        const textContentElement = document.createElement("p");
        if (typeof content === 'string') {
            textContentElement.textContent = content; // Set text if string
        } else if (content instanceof HTMLElement) {
             textContentElement.appendChild(content); // Append if already an element
        }
        messageBubble.appendChild(textContentElement);

        // *** CHANGE: Use append instead of prepend ***
        chatMessages.append(messageBubble);

        // *** CHANGE: Scroll to the bottom ***
        if (chatArea) {
            // Scroll the main scrollable container to its maximum height
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        return messageBubble; // Return the created element
    }

    // --- Keep the removeMessageById function as it is ---
    function removeMessageById(id) {
        const messageToRemove = document.getElementById(id);
        if (messageToRemove) {
            messageToRemove.remove();
        }
    }

    /**
     * Sends a text question to the backend and streams the response into a message bubble.
     * @param {string} question - The question text to send.
     */
    async function askQuestion(question) {
        setProcessingState(true); // Indicate processing
        displayMessage(question, "user-message"); // Show the user's question

        const thinkingStatusId = `thinking-${Date.now()}`; // Unique ID for temp message
        displayMessage("🤔 Assistant is thinking...", "system-message", thinkingStatusId);

        let assistantBubble = null; // Reference to the assistant's message bubble
        let contentElement = null; // Reference to the <p> tag inside the bubble
        let accumulatedAnswer = ""; // Store the full response text

        try {
            // API call to the question endpoint (ensure URL is correct)
            // Make sure '/api/ask_question/' matches your urls.py
            const response = await fetch('/api/ask_question/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF if needed
                },
                body: JSON.stringify({ question: question }),
            });

            // --- Handle non-OK HTTP responses BEFORE attempting to stream ---
            if (!response.ok) {
                removeMessageById(thinkingStatusId); // Remove "Thinking..." message
                let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                try {
                     // Try to get more specific error from backend JSON response
                     const err = await response.json();
                     errorMsg = err.error || errorMsg;
                } catch { /* Ignore if error response isn't JSON */ }
                throw new Error(errorMsg); // Throw error to be caught below
            }

            // --- Response is OK, prepare for streaming ---
            removeMessageById(thinkingStatusId); // Remove "Thinking..."
            assistantBubble = displayMessage("", "assistant-message"); // Create empty bubble
            contentElement = assistantBubble?.querySelector('p'); // Target the <p> tag

            // Ensure we have a target element before proceeding
            if (!contentElement) {
                throw new Error("Could not create assistant message bubble.");
            }

            const reader = response.body.getReader(); // Get stream reader
            const decoder = new TextDecoder(); // To decode UTF-8 stream
            let reading = true;

            // Read chunks from the stream
            while (reading) {
                const { done, value } = await reader.read();
                if (done) {
                    reading = false; // Exit loop when stream ends
                    break;
                }
                // Decode chunk and append to display
                const chunk = decoder.decode(value, { stream: true });
                accumulatedAnswer += chunk;
                contentElement.textContent = accumulatedAnswer; // Update the bubble content

                // Keep scrolling to show the latest text
                if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
            }

            // --- Stream finished ---
            // Optional: Apply Markdown/Sanitization to the complete response
            if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                 try {
                     const rawHtml = marked.parse(accumulatedAnswer); // Convert full text
                     contentElement.innerHTML = DOMPurify.sanitize(rawHtml); // Sanitize and set HTML
                     if (chatArea) chatArea.scrollTop = chatArea.scrollHeight; // Re-scroll after potentially large render
                 } catch(e) { console.error("Markdown/DOMPurify error on final answer:", e); }
            }

        } catch (error) { // Catch fetch errors or errors thrown from response handling
            console.error("Asking Question Error:", error);
            removeMessageById(thinkingStatusId); // Ensure "Thinking..." is removed
            // Display error message distinctively
            if (assistantBubble && contentElement) {
                // If stream started, append error info for context
                contentElement.textContent = accumulatedAnswer + `\n\n--- Error: ${error.message} ---`;
            } else {
                // If fetch failed before creating the bubble
                displayMessage(`❌ Error: ${error.message}`, "error-message");
            }
        } finally {
            setProcessingState(false); // Re-enable input fields
            console.log("Ask question processing finished.");
        }
    }

    // ========================
    // Sending Logic (Main Action)
    // ========================
    function sendMessage() {
        if (isProcessing || !messageInput) return; // Exit if busy or input missing

        const messageText = messageInput.value.trim();
        // Regex to detect YouTube URLs (various formats)
        const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
        const youtubeUrlMatch = messageText.match(youtubeUrlRegex);
        let questionForVideo = ""; // Store question associated with video/file

        // --- Determine Action Priority: File > URL > Text ---
        if (selectedFiles.length > 0) {
            // --- Action: Process File ---
            questionForVideo = messageText; // Assume any text is a question for the file
            const fileToProcess = selectedFiles[0]; // Process the first selected file

            // Display user's intent
            displayMessage(`Attaching file: ${fileToProcess.name}`, "user-message");
             if (questionForVideo) {
                 // Show the question that will be asked after processing
                 displayMessage(`Question: ${questionForVideo}`, "system-message");
             }

            // Prepare form data
            const formData = new FormData();
            formData.append('videoFile', fileToProcess); // Key expected by backend
             if (questionForVideo) {
                formData.append('question', questionForVideo); // Add question if present
             }

            processVideo(formData); // Call backend function

            // Clear inputs *after* initiating the process
            selectedFiles = [];
            if(fileInput) fileInput.value = ""; // Clear the actual file input element
            updateFilePreview();
            messageInput.value = "";
            messageInput.style.height = 'auto'; // Reset textarea height

        } else if (youtubeUrlMatch) {
            // --- Action: Process URL ---
            const videoUrl = youtubeUrlMatch[0];
            // Extract text that is *not* the URL as the question
            questionForVideo = messageText.replace(videoUrl, "").trim();

            displayMessage(`Video URL: ${videoUrl}`, "user-message");
             if (questionForVideo) {
                 displayMessage(`Question: ${questionForVideo}`, "system-message");
             }

            // Send URL and optional question as JSON
            processVideo({ videoUrl: videoUrl, question: questionForVideo });

            // Clear text input after initiating process
            messageInput.value = "";
            messageInput.style.height = 'auto';

        } else if (messageText) {
            // --- Action: Send Text Question ---
            askQuestion(messageText); // Handles displaying user message internally now
            // Clear text input after sending
            messageInput.value = "";
            messageInput.style.height = 'auto';
        } else {
            // --- Action: Nothing Entered ---
            displayMessage("Please enter a message, provide a Video URL, or attach a video file.", "system-message error-message"); // Use error class
        }
    }

    // Attach send logic to button and Enter key
    if (sendButton) {
        sendButton.addEventListener("click", sendMessage);
    }
    if (messageInput) {
        // Listen for Enter key (but not Shift+Enter for newlines)
        messageInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault(); // Prevent inserting a newline
                sendMessage();
            }
        });
    }

    // ========================
    // Backend Interaction Functions
    // ========================

    /**
     * Sends video file or URL (with optional question) to the backend for processing.
     * Handles the response, including potential initial answer.
     * @param {FormData|Object} videoData - Data to send (FormData or {videoUrl, question}).
     */
    function processVideo(videoData) {
        setProcessingState(true); // Indicate processing started
        const processingStatusId = `status-${Date.now()}`; // Unique ID for temporary message
        // Display temporary status message
        displayMessage("⏳ Uploading and processing video...", "system-message", processingStatusId);

        let fetchBody;
        let headers = {
             // 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF if needed
        };

        // Determine body type and headers based on input
        if (videoData instanceof FormData) {
            fetchBody = videoData; // Use FormData directly
            // Browser sets Content-Type with boundary for FormData automatically
        } else { // JSON for URL
            fetchBody = JSON.stringify(videoData);
            headers['Content-Type'] = 'application/json';
        }

        // API call to the upload endpoint (ensure URL is correct)
        // Make sure '/api/upload_video/' matches your urls.py
        fetch('/api/upload_video/', {
            method: 'POST',
            headers: headers,
            body: fetchBody,
        })
        .then(response => {
            removeMessageById(processingStatusId); // Remove temporary status message
            // Check for HTTP errors (e.g., 404, 500)
            if (!response.ok) {
                // Try to parse a JSON error message from the backend
                return response.json().then(err => {
                    throw new Error(err.error || `Server error: ${response.status}`);
                }).catch(() => { // Fallback if error response isn't JSON
                    throw new Error(`Server error: ${response.status} ${response.statusText}`);
                });
            }
            return response.json(); // Parse successful JSON response
        })
        .then(data => {
            // Check for application-level errors returned in the JSON data
            if (data.error) {
                throw new Error(data.error);
            }
            // Display backend's success message
            displayMessage(`✅ ${data.message || 'Video processed.'}`, "system-message");

            // **Handle potential initial answer returned by the backend**
            if (data.answer) {
                const assistantBubble = displayMessage(data.answer, "assistant-message");
                // Apply Markdown/Sanitization if libraries are present
                const contentElement = assistantBubble?.querySelector('p');
                if (contentElement && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                     try {
                         const rawHtml = marked.parse(data.answer); // Convert Markdown to HTML
                         contentElement.innerHTML = DOMPurify.sanitize(rawHtml); // Sanitize and set HTML
                          if (chatArea) chatArea.scrollTop = chatArea.scrollHeight; // Re-scroll after render
                     } catch(e) { console.error("Markdown/DOMPurify error on initial answer:", e); }
                }
            }
        })
        .catch(error => {
            // Catch fetch errors or errors thrown from .then blocks
            removeMessageById(processingStatusId); // Ensure status message removal on error
            console.error("Processing Error:", error);
            // Display error message to the user
            displayMessage(`❌ Error processing video: ${error.message}`, "error-message");
        })
        .finally(() => {
            setProcessingState(false); // Re-enable inputs regardless of outcome
        });
    }


    // ========================
    // Chat Sharing Function (Optional)
    // ========================
    function shareChat() {
        if (!chatMessages) return;
        let chatText = "";
        // Get message bubbles in visual order (reverse DOM order due to prepend)
        const messages = Array.from(chatMessages.children).reverse();

        messages.forEach(msg => {
            let role = "System"; // Default role
            if (msg.classList.contains("user-message")) role = "User";
            else if (msg.classList.contains("assistant-message")) role = "Assistant";
            else if (msg.classList.contains("error-message")) role = "Error";

            // Get text content, preferring the inner <p> tag
            const text = msg.querySelector("p")?.textContent || msg.textContent || "";
            chatText += `${role}: ${text.trim()}\n\n`; // Format with role and spacing
        });

        if (!chatText.trim()) {
            alert("No chat messages to share.");
            return;
        }

        // Use Clipboard API to copy text
        navigator.clipboard.writeText(chatText.trim())
            .then(() => alert("Chat copied to clipboard!"))
            .catch(err => {
                console.error("Failed to copy chat:", err);
                alert("Failed to copy chat. See console for details.");
            });
    }
     // Add event listener if the share button exists
     if (shareButton) {
         shareButton.addEventListener('click', shareChat);
     }

    // ========================
    // CSRF Token Helper (Only needed if NOT using @csrf_exempt in Django)
    // ========================
    /*
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    */

}); // End DOMContentLoaded