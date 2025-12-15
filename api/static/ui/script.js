/**
 * Frontend JavaScript for Chat Interface
 * Handles user input, file uploads, API calls, streaming AI responses,
 * optional related video search, and client-side history management.
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
    // const loadingIndicator = document.getElementById("loadingIndicator"); // Replaced by per-message spinner
    const shareButton = document.querySelector(".share-btn");
    const findOtherVideosCheckbox = document.getElementById("findOtherVideos"); // Checkbox for YT search

    let selectedFiles = []; // Array to hold selected file objects for preview
    let isProcessing = false; // Flag to prevent multiple submissions
    let currentVideoContext = false; // Flag to track if base video context is loaded

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
        // Spinner is now handled per message, not globally tied to send button icon
        if (sendButton) {
             // Keep the send icon consistent
             sendButton.innerHTML = '<span class="material-icons">↑</span>';
             // Disable/enable button visually
             sendButton.style.opacity = processing ? 0.5 : 1;
             sendButton.style.cursor = processing ? 'not-allowed' : 'pointer';
        }
    }
    setProcessingState(false);

    // ========================
    // Dynamic Text Area Expansion (Unchanged)
    // ========================
    if (messageInput) {
        messageInput.addEventListener("input", function () {
            this.style.height = "auto";
            const scrollHeight = this.scrollHeight;
            const maxHeight = 150; // Max height in pixels
            this.style.height = Math.min(scrollHeight, maxHeight) + "px";
        });
    }

    // ========================
    // File Handling Functions (Unchanged)
    // ========================
    function updateFilePreview() {
        if (!filePreviewContainer) return;
        filePreviewContainer.innerHTML = "";
        if(inputContainer) {
            inputContainer.style.paddingTop = selectedFiles.length > 0 ? "15px" : "12px"; // Adjust spacing
        }
        selectedFiles.forEach((file, index) => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");

            let icon = '📄'; // Default icon
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
                e.stopPropagation(); // Prevent potential parent clicks
                removeFile(index);
            };
            fileTag.appendChild(removeBtn);
            filePreviewContainer.appendChild(fileTag);
        });
    }

    function removeFile(indexToRemove) {
        selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
        // Update the actual file input element to reflect the change
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
            // Replace existing selection with the new one
            selectedFiles = [];
            newFiles.forEach(newFile => {
                // Optional: Add checks for file type or size here if needed
                 if (newFile.type.startsWith("video/") || newFile.type.startsWith("audio/")) { // Allow audio too?
                     selectedFiles.push(newFile);
                 } else {
                      console.warn(`Ignoring unsupported file type: ${newFile.type}`);
                      // Optionally display a message to the user
                 }
            });
             // Limit to one file for processing
            if (selectedFiles.length > 1) {
                alert("Please select only one video file at a time.");
                selectedFiles = [selectedFiles[0]]; // Keep only the first one
            }

            // Update the file input's internal list to match selection
            const dataTransfer = new DataTransfer();
            selectedFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;

            updateFilePreview();
        });
    }

    if (attachButton) {
        attachButton.addEventListener("click", function () {
            if (isProcessing || !fileInput) return; // Check if we should do nothing
            fileInput.click();                     // If the checks pass, trigger the file dialog
        });
    }

    // ========================
    // Chat Message Functions (MODIFIED)
    // ========================

    /**
     * Displays a message in the chat area.
     * @param {string | HTMLElement} content - Text content or an HTML element to display.
     * @param {string} className - CSS class for the message bubble (e.g., 'user-message', 'assistant-message').
     * @param {string | null} [id=null] - Optional ID for the message bubble element.
     * @param {boolean} [showSpinner=false] - If true, adds a loading spinner before the text.
     * @returns {HTMLElement | null} The created message bubble element or null if chatMessages doesn't exist.
     */
    function displayMessage(content, className, id = null, showSpinner = false) {
        if (!chatMessages) return null;

        const messageBubble = document.createElement("div");
        if (id) messageBubble.id = id;
        messageBubble.classList.add(className); // Add base class

        // Create container for spinner + text to allow flex alignment
        const contentWrapper = document.createElement("div");
        contentWrapper.style.display = "flex";
        contentWrapper.style.alignItems = "center";
        contentWrapper.style.gap = "8px"; // Space between spinner and text

        if (showSpinner) {
            const spinner = document.createElement("div");
            spinner.classList.add("spinner"); // Add your CSS class for spinner animation
            contentWrapper.appendChild(spinner);
        }

        const textContentElement = document.createElement("p");
        textContentElement.style.margin = "0"; // Remove default paragraph margin

        if (typeof content === 'string') {
            // Basic check, treat as text. Use Markdown/Sanitize later if needed.
            textContentElement.textContent = content;
        } else if (content instanceof HTMLElement) {
            // If content is already an element (like the related videos container)
            textContentElement.appendChild(content);
        }
        contentWrapper.appendChild(textContentElement);
        messageBubble.appendChild(contentWrapper);


        chatMessages.appendChild(messageBubble); // Append to bottom

        scrollToBottom(); // Scroll down
        return messageBubble;
    }

    function removeMessageById(id) {
        const messageToRemove = document.getElementById(id);
        if (messageToRemove) {
            messageToRemove.remove();
        }
    }

    // Helper to update message content (e.g., remove spinner, change text)
    function updateMessageContent(id, newContent, showSpinner = false) {
         const messageBubble = document.getElementById(id);
         if (!messageBubble) return;

         // Find the content wrapper (assuming it's the first child)
         const contentWrapper = messageBubble.firstChild;
         if (!contentWrapper) return;

         // Clear existing content
         contentWrapper.innerHTML = '';

         if (showSpinner) {
            const spinner = document.createElement("div");
            spinner.classList.add("spinner");
            contentWrapper.appendChild(spinner);
         }

         const textContentElement = document.createElement("p");
         textContentElement.style.margin = "0";
         if (typeof newContent === 'string') {
            textContentElement.textContent = newContent;
         } else if (newContent instanceof HTMLElement) {
            textContentElement.appendChild(newContent);
         }
          contentWrapper.appendChild(textContentElement);
    }


    function scrollToBottom() {
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }

    // --- Markdown and Sanitization (Optional, requires libraries) ---
    function renderMarkdown(element, text) {
        if (element && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            try {
                // Ensure configuration allows target="_blank" for links if needed
                const dirtyHtml = marked.parse(text);
                element.innerHTML = DOMPurify.sanitize(dirtyHtml, { ADD_ATTR: ['target'] });
            } catch (e) {
                console.error("Markdown/DOMPurify error:", e);
                element.textContent = text; // Fallback to text if parsing fails
            }
        } else if (element) {
             element.textContent = text; // Fallback if libraries aren't loaded
        }
    }


    /**
     * Sends a text question (with history and YT flag) to the backend and streams the response.
     * Updates client-side history on successful completion.
     * @param {string} question - The question text to send.
     */
    async function askQuestion(question) {
        setProcessingState(true);
        displayMessage(question, "user-message"); // Show user's question immediately

        // ** Add user question to history immediately **
        chatHistory.push({ role: "user", content: question });

        const thinkingStatusId = `thinking-${Date.now()}`;
        const thinkingMessage = displayMessage("Assistant is thinking...", "system-message", thinkingStatusId, true); // Show spinner

        let assistantBubble = null;
        let contentElement = null;
        let accumulatedAnswer = ""; // Store the full response text

        // Get YT search preference
        const findVideosEnabled = findOtherVideosCheckbox ? findOtherVideosCheckbox.checked : false;

        try {
            const response = await fetch('/api/ask_question/', { // Ensure URL matches urls.py
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 'X-CSRFToken': getCookie('csrftoken'), // Add if CSRF needed
                },
                body: JSON.stringify({
                    question: question,
                    history: chatHistory.slice(0, -1), // Send history *before* this question
                    find_other_videos: findVideosEnabled // Send the flag
                }),
            });

            // --- Handle potential errors BEFORE streaming ---
             if (!response.ok) {
                 removeMessageById(thinkingStatusId); // Remove "thinking" message
                 let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                 try {
                      // Try to get specific error from backend JSON response
                     const err = await response.json();
                     errorMsg = err.error || errorMsg;
                 } catch { /* Ignore if error response isn't JSON */ }

                 // Display error message in chat
                 displayMessage(`⚠️ ${errorMsg}`, "error-message");

                  // ** Remove the failed user turn from history **
                 if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
                      chatHistory.pop();
                 }
                 setProcessingState(false); // Re-enable input
                 return; // Stop processing on error
             }

            // --- Response OK, prepare for streaming ---
            removeMessageById(thinkingStatusId); // Remove "thinking" message

            assistantBubble = displayMessage("", "assistant-message"); // Create empty bubble for streaming
            contentElement = assistantBubble?.querySelector('p'); // Get the <p> tag inside

            if (!assistantBubble || !contentElement) {
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
                accumulatedAnswer += chunk;

                // Render potentially incomplete Markdown as text for now
                contentElement.textContent = accumulatedAnswer;

                scrollToBottom(); // Keep scrolled down
            }

            // --- Stream finished ---
            // Render final accumulated answer with Markdown/Sanitization
             renderMarkdown(contentElement, accumulatedAnswer);
             scrollToBottom(); // Ensure scrolled down after potential height change

            // ** Add successful assistant response to history **
            if (accumulatedAnswer && !accumulatedAnswer.includes("--- Error:")) {
                 // Check for the specific "cannot answer" message from backend
                 const cannotAnswerPattern = /This specific detail doesn.*?t seem to be covered.*?Should I answer using my general knowledge\? \(Yes\/No\)/i;
                 if (!cannotAnswerPattern.test(accumulatedAnswer)) {
                     // Only add to history if it's a "real" answer, not the prompt to use general knowledge
                     chatHistory.push({ role: "assistant", content: accumulatedAnswer });
                 } else {
                     // Optionally handle the "cannot answer" message differently if needed
                     console.log("Assistant indicated it couldn't answer from transcript.");
                 }
            } else if (!accumulatedAnswer) {
                // Handle case where stream finished but was empty
                contentElement.textContent = "(AI returned no content)";
                // Decide if empty responses should be added to history
                // chatHistory.push({ role: "assistant", content: "(AI returned no content)" });
            } else {
                 // Handle cases where the stream itself contained an error message from backend helper
                 console.error("Stream contained error:", accumulatedAnswer);
                 // Error is already displayed, don't add to history
            }


        } catch (error) {
            console.error("Asking Question Error:", error);
            removeMessageById(thinkingStatusId); // Ensure thinking removed if fetch failed early

            // Display error message
            const errorText = `⚠️ Error: ${error.message}`;
            if (assistantBubble && contentElement) {
                // If stream started but failed midway, update the bubble
                 contentElement.textContent = accumulatedAnswer + `\n\n${errorText}`;
            } else {
                // If fetch failed before creating the bubble
                displayMessage(errorText, "error-message");
            }

            // ** Remove the failed user turn from history **
            if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
                chatHistory.pop();
            }

        } finally {
            setProcessingState(false); // Re-enable input fields
            console.log("Ask question processing finished. History length:", chatHistory.length);
        }
    }

    // ========================
    // Sending Logic (Main Action - MODIFIED)
    // ========================
    function sendMessage() {
        if (isProcessing || !messageInput) return;

        let messageText = messageInput.value.trim();
        const isUrl = messageText.startsWith("http://") || messageText.startsWith("https://");
        const findVideosEnabled = findOtherVideosCheckbox ? findOtherVideosCheckbox.checked : false;

        // --- Determine Action Priority: File > URL > Text ---
        if (selectedFiles.length > 0) {
            // --- Action: Process File ---
            console.log("New file upload detected, clearing client history.");
            chatHistory = []; // Clear history for new context
            currentVideoContext = false; // Mark context as not ready yet

            const questionForVideo = messageText; // Assume any text is question for the file
            const fileToProcess = selectedFiles[0]; // Process only the first file

            displayMessage(`Attaching file: ${fileToProcess.name}`, "user-message");
            if (questionForVideo) {
                // Display initial question separately if provided
                 displayMessage(`Initial question: "${questionForVideo}"`, "system-message");
            }

            const formData = new FormData();
            formData.append('videoFile', fileToProcess);
            if (questionForVideo) {
                formData.append('question', questionForVideo);
            }
            // Add the YouTube search flag
             formData.append('find_other_videos', findVideosEnabled);


            processVideo(formData, questionForVideo); // Pass question separately for potential history init

            // Clear inputs *after* initiating the process
            selectedFiles = [];
            if(fileInput) fileInput.value = ""; // Clear the file input
            updateFilePreview();
            messageInput.value = "";
            messageInput.style.height = 'auto'; // Reset height

        } else if (isUrl) {
             // --- Action: Process URL ---
             console.log("URL detected, clearing client history.");
             chatHistory = [];
             currentVideoContext = false;

             const videoUrl = messageText;
             const questionForVideo = ""; // Don't treat URL itself as question

             displayMessage(`Processing URL: ${videoUrl}`, "user-message");

             // Send URL and flag as JSON
             const jsonData = {
                 videoUrl: videoUrl,
                 question: questionForVideo,
                 find_other_videos: findVideosEnabled // Add the flag
             };

             processVideo(jsonData, questionForVideo);

             messageInput.value = "";
             messageInput.style.height = 'auto';

        } else if (messageText) {
            // --- Action: Send Text Question ---
             if (!currentVideoContext) {
                  displayMessage("⚠️ Please upload a video or provide a URL first before asking questions.", "error-message");
                  return; // Don't send question if no context is loaded
             }
            askQuestion(messageText);
            messageInput.value = "";
            messageInput.style.height = 'auto';
        } else {
            // --- Action: Nothing Entered ---
            displayMessage("⚠️ Please enter a message, provide a Video URL, or attach a video file.", "error-message");
        }
    }

    // Attach send logic to button and Enter key
    if (sendButton) sendButton.addEventListener("click", sendMessage);
    if (messageInput) {
        messageInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault(); // Prevent newline on Enter
                sendMessage();
            }
        });
    }

    // ========================
    // Backend Interaction Function (MODIFIED)
    // ========================
    /**
     * Sends video file or URL (with optional initial question and YT flag) to the backend.
     * Handles status updates and displays results including related videos.
     * Initializes client-side history if an initial answer is received.
     * @param {FormData|Object} videoData - Data to send (FormData or {videoUrl, question, find_other_videos}).
     * @param {string} [initialQuestion=''] - The initial question asked, for history init.
     */
    async function processVideo(videoData, initialQuestion = '') {
        setProcessingState(true);
        currentVideoContext = false; // Context is processing, not ready

        const processingStatusId = `status-${Date.now()}`;
        let statusMessage = displayMessage(" Uploading & Initializing...", "system-message", processingStatusId, true); // Show spinner

        let fetchBody;
        let headers = { /* 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF if needed */ };

        if (videoData instanceof FormData) {
            fetchBody = videoData;
            // Content-Type is set automatically by browser for FormData
        } else {
            // Assuming videoData is a JS object for URL processing
            fetchBody = JSON.stringify(videoData);
            headers['Content-Type'] = 'application/json';
        }

        try {
            // Step 1: Send request
            const response = await fetch('/api/upload_video/', { // Ensure URL matches urls.py
                method: 'POST',
                headers: headers,
                body: fetchBody,
            });

             // Update status message (can add more specific steps later if backend provides them)
             updateMessageContent(processingStatusId, " Processing video (transcribing, analyzing...).", true);


            if (!response.ok) {
                // Try to get error message from backend JSON
                let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                 try {
                     const err = await response.json();
                     errorMsg = err.error || errorMsg;
                 } catch { /* Ignore if error response isn't JSON */ }
                throw new Error(errorMsg); // Throw error to be caught below
            }

            // Step 2: Process successful response
            const data = await response.json();

            if (data.error) { // Check for application-level errors from backend
                 throw new Error(data.error);
            }

            // Processing finished - remove spinner from status message
             updateMessageContent(processingStatusId, `✅ ${data.message || 'Video processed.'}`, false);
             currentVideoContext = true; // Mark context as ready

            // ** Display Related YouTube Videos (if provided) **
            if (data.youtube_videos && data.youtube_videos.length > 0) {
                const linksContainer = document.createElement('div');
                linksContainer.classList.add('related-videos-container');
                const titleP = document.createElement('p');
                titleP.textContent = 'Related YouTube Videos Found:';
                titleP.style.fontWeight = 'bold';
                linksContainer.appendChild(titleP);

                const linksList = document.createElement('ul');
                 linksList.style.listStyle = 'disc'; // Use standard bullets
                 linksList.style.paddingLeft = '20px'; // Indent list

                data.youtube_videos.forEach(video => {
                    const listItem = document.createElement('li');
                     listItem.style.marginBottom = '5px'; // Space between items
                    const link = document.createElement('a');
                    link.href = video.url;
                    link.textContent = video.title;
                    link.target = '_blank'; // Open in new tab
                    link.rel = 'noopener noreferrer';
                    listItem.appendChild(link);
                    linksList.appendChild(listItem);
                });
                linksContainer.appendChild(linksList);
                 // Display this container as a new message
                 displayMessage(linksContainer, "system-message");
            }

            // ** Initialize history if initial question was asked AND answered **
            if (initialQuestion && data.answer) {
                 chatHistory = [
                     { role: 'user', content: initialQuestion },
                     { role: 'assistant', content: data.answer }
                 ];
                 console.log("Initialized history with initial Q&A.");
                 // Display the initial answer properly
                 const assistantBubble = displayMessage("", "assistant-message");
                 const contentElement = assistantBubble?.querySelector('p');
                 renderMarkdown(contentElement, data.answer); // Use Markdown renderer

            } else {
                 // If no initial question or no answer came back, history remains empty
                 chatHistory = [];
                 console.log("No initial Q&A, history is empty.");
            }
             scrollToBottom(); // Scroll after potentially adding messages

        } catch (error) {
            // Handle fetch errors or errors thrown from response processing
            console.error("Processing Error:", error);
            // Remove or update the status message to show the error
            updateMessageContent(processingStatusId, `⚠️ Error processing video: ${error.message}`, false); // Show error, remove spinner
            currentVideoContext = false; // Context failed to load
            chatHistory = []; // Clear history on error too
        } finally {
            setProcessingState(false); // Re-enable inputs regardless of success/failure
        }
    }

    // ========================
    // Chat Sharing Function (Unchanged)
    // ========================
    function shareChat() {
        if (chatHistory.length === 0) {
            alert("No chat history to share.");
            return;
        }
        let chatText = "Chat History:\n\n";
        chatHistory.forEach(msg => {
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
    // CSRF Token Helper (If needed - Uncomment and adapt if using Django's CSRF)
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