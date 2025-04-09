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
   const chatArea = document.querySelector(".chat-area"); // The scrollable container
   const inputContainer = document.querySelector(".input-container");
   const loadingIndicator = document.getElementById("loadingIndicator"); // Spinner next to buttons
   const shareButton = document.querySelector(".share-btn"); // Optional share button

   // NOTE: The "analyzeFullTranscript" checkbox logic is kept from your previous code.
   // Remove if not needed.
   const analyzeFullTranscriptCheckbox = document.getElementById("analyzeFullTranscript");

   let selectedFiles = []; // Array to hold selected file objects for preview
   let isProcessing = false; // Flag to prevent multiple submissions
   let currentVideoContext = false; // Flag to track if video context is loaded

   // ** Client-side conversation history **
   let chatHistory = []; // Stores { role: 'user'/'assistant', content: '...' }

   // ========================
   // UI State Management (MODIFIED)
   // ========================
   function setProcessingState(processing) {
       isProcessing = processing;
       // ** Keep messageInput ALWAYS enabled **
       if (messageInput) messageInput.disabled = false; // No longer disabling the text area

       // ** ONLY disable the buttons **
       if (sendButton) sendButton.disabled = processing;
       if (attachButton) attachButton.disabled = processing;

       // Show/hide loading indicator (spinner)
       if (loadingIndicator) {
           loadingIndicator.style.display = processing ? "inline-block" : "none";
       }

       // Optionally change send button icon/text (e.g., hide arrow when loading)
       if (sendButton) {
            if (processing) {
               // Optional: Change icon during processing, e.g., show spinner INSIDE button
               // sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
            } else {
                sendButton.innerHTML = '<span class="material-icons">↑</span>'; // Restore arrow icon
            }
       }
   }
   setProcessingState(false); // Initialize state

   // ========================
   // Dynamic Text Area Expansion
   // ========================
   if (messageInput) {
       messageInput.addEventListener("input", function () {
           this.style.height = "auto"; // Reset height
           const scrollHeight = this.scrollHeight;
           const maxHeight = 150; // Max height in pixels before scrolling appears in the textarea itself
           this.style.height = Math.min(scrollHeight, maxHeight) + "px";
       });
   }

   // ========================
   // File Handling Functions (Unchanged - Assumed Correct)
   // ========================
   function updateFilePreview() {
       if (!filePreviewContainer) return;

       filePreviewContainer.innerHTML = "";
       if(inputContainer) {
           inputContainer.style.paddingTop = selectedFiles.length > 0 ? "15px" : "12px";
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
           const maxLen = 25; // Max filename length before ellipsis
           fileName.textContent = file.name.length > maxLen
               ? file.name.substring(0, maxLen - 3) + '...'
               : file.name;
           fileName.title = file.name; // Show full name on hover
           fileTag.appendChild(fileName);

           const removeBtn = document.createElement("button");
           removeBtn.textContent = "✖";
           removeBtn.classList.add("remove-file-btn");
           // Simple styling, ideally use CSS classes
           removeBtn.style.background = 'none';
           removeBtn.style.border = 'none';
           removeBtn.style.color = '#888';
           removeBtn.style.marginLeft = '10px';
           removeBtn.style.cursor = 'pointer';
           removeBtn.style.fontSize = '14px';
           removeBtn.setAttribute('aria-label', `Remove ${file.name}`);

           removeBtn.onclick = (e) => {
               e.stopPropagation(); // Prevent potential parent event handlers
               removeFile(index);
           };
           fileTag.appendChild(removeBtn);
           filePreviewContainer.appendChild(fileTag);
       });
   }

   function removeFile(indexToRemove) {
       selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
       // Update the actual file input's file list to match the preview
       const dataTransfer = new DataTransfer();
       selectedFiles.forEach(file => dataTransfer.items.add(file));
       if (fileInput) {
           fileInput.files = dataTransfer.files;
       }
       updateFilePreview(); // Refresh the preview
   }

   if (fileInput) {
       fileInput.addEventListener("change", function (event) {
           const newFiles = Array.from(event.target.files);
           // Replace existing selection with new ones
           selectedFiles = [];
           newFiles.forEach(newFile => {
               // Optional: Add checks for file type or size here
               selectedFiles.push(newFile);
           });

           // Update the file input's internal list (important for form submission if not using fetch)
           const dataTransfer = new DataTransfer();
           selectedFiles.forEach(file => dataTransfer.items.add(file));
           fileInput.files = dataTransfer.files;

           updateFilePreview(); // Show the new selection
       });
   }

   if (attachButton) {
       attachButton.addEventListener("click", function () {
           if (isProcessing || !fileInput) return; // Don't allow if processing or input missing
           fileInput.click(); // Trigger hidden file input click
       });
   }

   // ========================
   // Chat Message Functions (REVISED with Conditional Scrolling)
   // ========================
   /**
    * Displays a message in the chat area with conditional scrolling.
    * @param {string | HTMLElement} content - Text content or an HTML element.
    * @param {string} className - CSS class for the message bubble (e.g., 'user-message').
    * @param {string | null} [id=null] - Optional ID for the message bubble.
    * @returns {HTMLElement | null} The created message bubble element or null.
    */
   function displayMessage(content, className, id = null) {
       // Ensure necessary elements exist
       if (!chatMessages || !chatArea) {
           console.error("Chat message container or scroll area not found.");
           return null;
       }

       // --- Conditional Scroll Logic ---
       const tolerance = 10; // Pixels tolerance - adjust if needed
       const isScrolledToBottom = chatArea.scrollHeight - chatArea.clientHeight <= chatArea.scrollTop + tolerance;
       // --- End Conditional Scroll Logic Check ---

       const messageBubble = document.createElement("div");
       if (id) messageBubble.id = id;
       messageBubble.classList.add("message-bubble"); // General class for styling
       messageBubble.classList.add(className); // Specific class (user/assistant/system)

       // Assuming text content goes into a <p> tag for styling and semantic reasons
       const textContentElement = document.createElement("p");

       // Handle content (simple text or existing element)
       // NOTE: Raw HTML is NOT inserted here for security. Markdown rendering happens later.
       if (typeof content === 'string') {
           textContentElement.textContent = content; // Safely sets text content
       } else if (content instanceof HTMLElement) {
           textContentElement.innerHTML = ''; // Clear potential old content
           textContentElement.appendChild(content); // Append if it's already an element
       }
       messageBubble.appendChild(textContentElement);

       chatMessages.append(messageBubble); // Append the new message bubble to the container

       // --- Conditional Scroll Logic Action ---
       // Scroll down automatically ONLY if the user was already near the bottom
       if (isScrolledToBottom) {
           // Use timeout to ensure scrolling happens after rendering is complete
            setTimeout(() => { chatArea.scrollTop = chatArea.scrollHeight; }, 0);
       }
       // --- End Conditional Scroll Logic Action ---

       return messageBubble; // Return the created bubble element
   }

   /**
    * Removes a message bubble by its ID.
    * @param {string} id - The ID of the message element to remove.
    */
   function removeMessageById(id) {
       const messageToRemove = document.getElementById(id);
       if (messageToRemove) {
           messageToRemove.remove();
       }
   }

   /**
    * Sends a text question (with history) to the backend and streams the response.
    * Updates client-side history and renders Markdown response.
    * Uses conditional scrolling during streaming.
    * @param {string} question - The question text to send.
    */
   async function askQuestion(question) {
       setProcessingState(true); // Disable buttons, show loader
       displayMessage(question, "user-message"); // Show user question (uses conditional scroll)
       // ** Add user question to history immediately **
       chatHistory.push({ role: "user", content: question });

       const thinkingStatusId = `thinking-${Date.now()}`;
       displayMessage("🤔 Assistant is thinking...", "system-message", thinkingStatusId); // Uses conditional scroll

       let assistantBubble = null;
       let contentElement = null; // Will hold the <p> tag inside the bubble
       let accumulatedAnswer = ""; // Store the full response text

       try {
           // API call, including client-side history
           const response = await fetch('/api/ask_question/', { // Ensure URL matches Django urls.py
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json',
                   // 'X-CSRFToken': getCookie('csrftoken'), // Add if CSRF protection is enabled server-side
               },
               // Send current question AND history (excluding the question just added)
               body: JSON.stringify({
                   question: question,
                   history: chatHistory.slice(0, -1) // Send history *before* this question
               }),
           });

           removeMessageById(thinkingStatusId); // Remove "Thinking..." message

           if (!response.ok || !response.body) { // Check if response body exists for streaming
               let errorMsg = `Server error: ${response.status} ${response.statusText}`;
               try {
                    const err = await response.json(); // Try to get JSON error from body
                    errorMsg = err.error || errorMsg;
               } catch { /* Ignore if error response isn't JSON */ }
               throw new Error(errorMsg);
           }

           // --- Response OK, prepare for streaming ---
           assistantBubble = displayMessage("", "assistant-message"); // Create empty bubble (uses conditional scroll)
           contentElement = assistantBubble?.querySelector('p'); // Get the <p> tag

           if (!contentElement) {
               throw new Error("Could not create assistant message content element.");
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

               // --- Conditional Scroll Logic during Streaming ---
               const isScrolledToBottomLoop = chatArea
                   ? chatArea.scrollHeight - chatArea.clientHeight <= chatArea.scrollTop + 10
                   : true;

               // Update display with plain text during streaming for responsiveness
               contentElement.textContent = accumulatedAnswer;

               // Scroll down ONLY if user was already near the bottom
               if (isScrolledToBottomLoop && chatArea) {
                    // Use timeout for reliability after DOM update
                    setTimeout(() => { chatArea.scrollTop = chatArea.scrollHeight; }, 0);
               }
                // --- End Conditional Scroll Logic during Streaming ---
           }

           // --- Stream finished ---
           // ** Add successful assistant response to history **
           if (accumulatedAnswer && !accumulatedAnswer.includes("--- Error:")) {
                chatHistory.push({ role: "assistant", content: accumulatedAnswer });
           } else if (!accumulatedAnswer) {
                // Handle cases where stream finished but was empty
                chatHistory.push({ role: "assistant", content: "(AI returned no content)" });
           } // Don't add backend error messages to history as assistant turns

           // ** Apply Markdown Formatting to the FINAL complete response **
           // Requires marked.js and DOMPurify libraries to be included in your HTML:
           // <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
           // <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
           if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                try {
                    // Check scroll position BEFORE potentially changing height with formatted HTML
                    const isScrolledToBottomFinal = chatArea
                       ? chatArea.scrollHeight - chatArea.clientHeight <= chatArea.scrollTop + 10
                       : true;

                    // Convert potentially unsafe Markdown to safe HTML
                    const rawHtml = marked.parse(accumulatedAnswer);
                    contentElement.innerHTML = DOMPurify.sanitize(rawHtml); // Render safe HTML

                    // Scroll down conditionally AFTER rendering final HTML
                    if (isScrolledToBottomFinal && chatArea) {
                        setTimeout(() => { chatArea.scrollTop = chatArea.scrollHeight; }, 0);
                    }
                } catch(e) {
                    console.error("Markdown/DOMPurify error on final answer:", e);
                    // Keep plain text content if formatting fails
                    contentElement.textContent = accumulatedAnswer;
                }
           } else {
               console.warn("Markdown/DOMPurify libraries not found. Skipping formatting.");
               // If libraries aren't present, the plain text is already displayed
           }

       } catch (error) {
           console.error("Asking Question Error:", error);
           removeMessageById(thinkingStatusId); // Ensure thinking removed if it wasn't already

            // Display error message in the UI
            if (assistantBubble && contentElement) {
                // If stream started, show error appended to whatever was received
                contentElement.textContent = accumulatedAnswer + `\n\n--- Error: ${error.message} ---`;
                assistantBubble.classList.add("error-message"); // Style the bubble as an error
            } else {
                // If fetch failed before creating the bubble, show separate error message
                displayMessage(`❌ Error: ${error.message}`, "error-message");
            }

            // Remove the user message that caused the error from history to prevent resending
            if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
                chatHistory.pop();
            }
       } finally {
           setProcessingState(false); // Re-enable buttons
           console.log("Ask question processing finished. History length:", chatHistory.length);
       }
   }

   // ========================
   // Sending Logic (Main Action - Modified)
   // ========================
   function sendMessage() {
       // Check only 'isProcessing' which covers button state; input area is always enabled
       if (isProcessing || !messageInput) return;

       let messageText = messageInput.value.trim();
       const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
       const youtubeUrlMatch = messageText.match(youtubeUrlRegex);
       let questionForVideo = "";

       // --- Determine Action Priority: File > URL > Text ---
       if (selectedFiles.length > 0) {
           // --- Action: Process File ---
           console.log("New file upload detected, clearing client history.");
           chatHistory = []; // Clear history for new context
           currentVideoContext = false;

           questionForVideo = messageText; // Text entered alongside file is initial question
           const fileToProcess = selectedFiles[0]; // Process only the first file

           displayMessage(`Attaching file: ${fileToProcess.name}`, "user-message");
            if (questionForVideo) {
                displayMessage(`Initial question: ${questionForVideo}`, "system-message");
            }

           const formData = new FormData();
           formData.append('videoFile', fileToProcess);
            if (questionForVideo) {
                formData.append('question', questionForVideo);
            }

           processVideo(formData, questionForVideo); // Start backend processing

           // Clear inputs after initiating
           selectedFiles = [];
           if(fileInput) fileInput.value = ""; // Clear file input
           updateFilePreview();
           messageInput.value = ""; // Clear text input
           messageInput.style.height = 'auto'; // Reset height

       } else if (youtubeUrlMatch) { // Prioritize YouTube URLs if specifically matched
            // --- Action: Process YouTube URL ---
            console.log("YouTube URL detected, clearing client history.");
            chatHistory = [];
            currentVideoContext = false;

            const videoUrl = youtubeUrlMatch[0];
            // Text after the URL is the initial question
            questionForVideo = messageText.replace(videoUrl, "").trim();

            displayMessage(`Processing YouTube URL: ${videoUrl}`, "user-message");
            if (questionForVideo) {
                displayMessage(`Initial question: ${questionForVideo}`, "system-message");
            }

            // Send URL and optional question as JSON
            processVideo({ videoUrl: videoUrl, question: questionForVideo }, questionForVideo);

            messageInput.value = "";
            messageInput.style.height = 'auto';

       } else if (messageText.startsWith("http://") || messageText.startsWith("https://")) {
            // --- Action: Process Any Other URL ---
            console.log("Generic URL detected, clearing client history.");
            chatHistory = [];
            currentVideoContext = false;

            // Treat the whole input as the URL, no separate question extracted here
            const genericUrl = messageText;
            questionForVideo = "";
            displayMessage(`Processing URL: ${genericUrl}`, "user-message");

            processVideo({ videoUrl: genericUrl, question: "" }, ""); // Send URL

            messageInput.value = "";
            messageInput.style.height = 'auto';

       } else if (messageText) {
           // --- Action: Ask Question (Text Only) ---
           let questionToSend = messageText;
           // Example: Prefix question if checkbox is checked (kept from your previous code)
           if (analyzeFullTranscriptCheckbox && analyzeFullTranscriptCheckbox.checked) {
               questionToSend = "analyze full transcript " + messageText;
               console.log("Prefixed question with 'analyze full transcript'");
           }
           askQuestion(questionToSend); // Send text question to backend
           messageInput.value = ""; // Clear input after sending
           messageInput.style.height = 'auto'; // Reset height
       } else {
           // --- Action: Nothing Entered ---
           // Optionally display a temporary message or do nothing
           displayMessage("Please enter a message, provide a Video URL, or attach a video file.", "system-message error-message");
           // Remove the warning after a delay
           setTimeout(() => {
               const warnings = document.querySelectorAll('.system-message.error-message');
               warnings.forEach(w => {
                   if (w.textContent.includes("Please enter a message")) w.remove();
               });
           }, 3000);
       }
   }

   // Attach send logic to button click and Enter key press
   if (sendButton) sendButton.addEventListener("click", sendMessage);
   if (messageInput) {
       messageInput.addEventListener("keydown", function (event) {
           // Send on Enter, but allow Shift+Enter for new line
           if (event.key === "Enter" && !event.shiftKey) {
               event.preventDefault(); // Prevent default Enter behavior (new line)
               sendMessage();
           }
       });
   }

   // ========================
   // Backend Interaction - Video Upload (MODIFIED for History Init)
   // ========================
   /**
    * Sends video file or URL to the backend /api/upload_video/.
    * Handles JSON response, including potential initial answer.
    * Initializes client-side history based on response.
    * @param {FormData|Object} videoData - Data to send (FormData or {videoUrl, question}).
    * @param {string} [initialQuestion=''] - The initial question asked by the user.
    */
   function processVideo(videoData, initialQuestion = '') {
       setProcessingState(true); // Disable buttons, show loader
       currentVideoContext = false; // Mark context as not ready during processing
       const processingStatusId = `status-${Date.now()}`;
       // Uses conditional scroll
       displayMessage("⏳ Uploading and processing video...", "system-message", processingStatusId);

       let fetchBody;
       let headers = { /* 'X-CSRFToken': getCookie('csrftoken'), // Add if needed */ };

       if (videoData instanceof FormData) {
           // FormData sets its own Content-Type header
           fetchBody = videoData;
       } else {
           // Plain object, send as JSON
           fetchBody = JSON.stringify(videoData);
           headers['Content-Type'] = 'application/json';
       }

       // API call to upload endpoint
       fetch('/api/upload_video/', { // Ensure URL matches Django urls.py
           method: 'POST',
           headers: headers,
           body: fetchBody,
       })
       .then(response => {
           // Always remove processing status message
           removeMessageById(processingStatusId);
           if (!response.ok) {
               // Try to parse JSON error first, fallback to status text
               return response.json()
                   .then(err => { throw new Error(err.error || `Server error: ${response.status}`); })
                   .catch(() => { throw new Error(`Server error: ${response.status} ${response.statusText}`); });
           }
           return response.json(); // Parse successful JSON response
       })
       .then(data => {
           if (data.error) { // Check for application-level errors in JSON
               throw new Error(data.error);
           }

           // Uses conditional scroll
           displayMessage(`✅ ${data.message || 'Video processed successfully.'}`, "system-message");
           currentVideoContext = true; // Mark context as ready for questions

           // ** Initialize client history ONLY if initial question was asked AND answered **
           if (initialQuestion && data.answer) {
                chatHistory = [
                    { role: 'user', content: initialQuestion },
                    { role: 'assistant', content: data.answer }
                ];
                console.log("Initialized client history with initial Q&A.");

                // Display the initial answer using displayMessage (handles scroll)
                const assistantBubble = displayMessage(data.answer, "assistant-message");

                // Apply Markdown to the initial answer if libraries are available
                const contentElement = assistantBubble?.querySelector('p');
                if (contentElement && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                     try {
                        // Check scroll before final render
                        const isScrolledToBottomFinal = chatArea
                           ? chatArea.scrollHeight - chatArea.clientHeight <= chatArea.scrollTop + 10
                           : true;

                        const rawHtml = marked.parse(data.answer);
                        contentElement.innerHTML = DOMPurify.sanitize(rawHtml);

                        // Conditional scroll after render
                        if (isScrolledToBottomFinal && chatArea) {
                            setTimeout(() => { chatArea.scrollTop = chatArea.scrollHeight; }, 0);
                        }
                     } catch(e) { console.error("Markdown/DOMPurify error on initial answer:", e); }
                }
           } else {
                // If no initial question was asked, or if it was asked but backend didn't provide 'answer'
                chatHistory = []; // Start with empty history
                console.log("No initial Q&A provided by backend, history is empty.");
           }
       })
       .catch(error => {
           removeMessageById(processingStatusId); // Ensure status removed on error
           console.error("Video Processing Error:", error);
           // Uses conditional scroll
           displayMessage(`❌ Error processing video: ${error.message}`, "error-message");
           currentVideoContext = false; // Context failed
           chatHistory = []; // Clear history on processing error
       })
       .finally(() => {
           setProcessingState(false); // Re-enable buttons
       });
   }

   // ========================
   // Chat Sharing Function (Optional - Uses client history)
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

       // Use Clipboard API
       navigator.clipboard.writeText(chatText.trim())
           .then(() => alert("Chat history copied to clipboard!"))
           .catch(err => {
               console.error("Failed to copy chat:", err);
               alert("Failed to copy chat. See console for details.");
           });
   }
    // Attach to button if it exists
    if (shareButton) {
        shareButton.addEventListener('click', shareChat);
    }

   // ========================
   // CSRF Token Helper (Include if needed for non-GET requests in Django)
   // ========================
   /* function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
   } */

   // ========================
   // Initial UI setup or Welcome Message (Optional)
   // ========================
   // displayMessage("Welcome! Upload a video or provide a URL to get started.", "system-message");

}); // End DOMContentLoaded