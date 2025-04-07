document.addEventListener("DOMContentLoaded", function () {
    // ========================
    // DOM Element Selection
    // ========================
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const filePreviewContainer = document.getElementById("filePreviewContainer");
    const chatMessages = document.getElementById("chatMessages"); // The scrollable container
    const signinButton = document.getElementById("signin-button");
    const shareButton = document.querySelector(".share-btn");
    const inputContainer = document.querySelector(".input-container"); // Overall input section
    const loadingIndicator = document.getElementById("loadingIndicator"); // Spinner element

    let selectedFiles = [];
    let isProcessing = false; // Flag to manage processing state

    // ========================
    // UI State Management
    // ========================
    function setProcessingState(processing) {
        isProcessing = processing;
        sendButton.disabled = processing; // Disable/enable send button
        attachButton.disabled = processing; // Disable/enable attach button

        if (loadingIndicator) {
            // Show/hide the spinner element
            loadingIndicator.style.display = processing ? "inline-block" : "none";
        }

        // Optional: Clear button text change if icon is preferred
        // sendButton.textContent = processing ? "..." : "↑";
        // If you keep the text change, ensure it reverts correctly
        if (!processing) {
             sendButton.innerHTML = '<span class="material-icons">↑</span>'; // Or just '↑' if not using Material Icons span
        } else {
             // You might just rely on the disabled state and spinner
             // sendButton.textContent = '...';
        }
    }

    // Initialize UI state
    setProcessingState(false);

    // ========================
    // Dynamic Text Area Expansion
    // ========================
    messageInput.addEventListener("input", function () {
        this.style.height = "auto"; // Reset height to auto first
        const scrollHeight = this.scrollHeight;
        const maxHeight = 150; // Match CSS max-height

        // Set height based on content, up to the max height
        this.style.height = Math.min(scrollHeight, maxHeight) + "px";

        // Optional: If content exceeds max-height, ensure scrollbar is visible
        // this.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden'; (CSS handles this with overflow: auto)
    });


    // ========================
    // File Handling Functions
    // ========================

    /**
     * Updates the file preview section with selected files.
     */
    function updateFilePreview() {
        filePreviewContainer.innerHTML = ""; // Clear previous previews
        // Adjust container padding/min-height slightly if files are present
        inputContainer.style.paddingTop = selectedFiles.length > 0 ? "15px" : "12px";

        selectedFiles.forEach((file, index) => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");

            // Basic preview for images
            if (file.type.startsWith("image/")) {
                const img = document.createElement("img");
                img.src = URL.createObjectURL(file);
                img.onload = () => URL.revokeObjectURL(img.src); // Clean up blob URL
                // Styles moved to CSS (.file-tag img)
                fileTag.appendChild(img);
            } else {
                // Placeholder for non-image files (e.g., an icon)
                const icon = document.createElement("span");
                icon.textContent = '📄'; // Simple file icon
                // icon.style.marginRight = "10px"; // Use gap in CSS
                fileTag.appendChild(icon);
            }

            const fileName = document.createElement("span");
            fileName.textContent = file.name.length > 25 ? file.name.substring(0, 22) + '...' : file.name; // Truncate long names
            fileTag.appendChild(fileName);

            // Add a remove button for each file
            const removeBtn = document.createElement("button");
            removeBtn.textContent = "✖";
            removeBtn.classList.add("remove-file-btn"); // Use class for styling
            removeBtn.onclick = (e) => {
                e.stopPropagation(); // Prevent click from propagating
                removeFile(index);
            };
            fileTag.appendChild(removeBtn);

            filePreviewContainer.appendChild(fileTag);
        });

         // Adjust overall input container min-height (optional)
        // inputContainer.style.minHeight = selectedFiles.length > 0 ? "100px" : "auto";
    }

    /**
     * Removes a file from the selection by index.
     */
    function removeFile(indexToRemove) {
        selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
        // Create a new DataTransfer object to update the file input's files list
        const dataTransfer = new DataTransfer();
        selectedFiles.forEach(file => dataTransfer.items.add(file));
        fileInput.files = dataTransfer.files; // Update the input
        updateFilePreview(); // Refresh the preview
    }


    /**
     * Handles file selection; updates the internal list and preview.
     */
    fileInput.addEventListener("change", function (event) {
        // Add newly selected files to the existing list
        const newFiles = Array.from(event.target.files);
        // Prevent duplicates (optional, based on name and size)
        newFiles.forEach(newFile => {
            if (!selectedFiles.some(existingFile => existingFile.name === newFile.name && existingFile.size === newFile.size)) {
                selectedFiles.push(newFile);
            }
        });

        // Update the file input's internal list to reflect the combined selection
        const dataTransfer = new DataTransfer();
        selectedFiles.forEach(file => dataTransfer.items.add(file));
        fileInput.files = dataTransfer.files;

        updateFilePreview();
    });

    // Opens the file manager when the attach button is clicked
    attachButton.addEventListener("click", function () {
        if (isProcessing) return; // Don't allow attaching while processing
        fileInput.click();
    });

    // ========================
    // Chat Message Functions
    // ========================

    /**
     * Appends a message bubble to the chat area and scrolls down.
     */
    function displayMessage(messageOrHtml, className) {
        const messageBubble = document.createElement("div");
        messageBubble.classList.add(className);
    
        if (className === 'assistant-message') {
            if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                 const rawHtml = marked.parse(messageOrHtml); // messageOrHtml is the full answer string
                 messageBubble.innerHTML = DOMPurify.sanitize(rawHtml);
            } else {
                 const textNode = document.createElement("p");
                 textNode.textContent = messageOrHtml; // Fallback
                 messageBubble.appendChild(textNode);
            }
        } else {
            const textNode = document.createElement("p");
            textNode.textContent = messageOrHtml;
            messageBubble.appendChild(textNode);
        }
    
        chatMessages.prepend(messageBubble); // Prepend for top display
        const scrollContainer = document.querySelector('.chat-area');
        if(scrollContainer) scrollContainer.scrollTop = 0; // Scroll to top
    }

    /**
     * Handles the primary action of sending messages, URLs, or files.
     */
    function sendMessage() {
        if (isProcessing) return; // Prevent multiple submissions

        const messageText = messageInput.value.trim();
        // Slightly more robust YouTube Regex (handles various URL formats including shorts)
        const youtubeUrlRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
        const youtubeUrlMatch = messageText.match(youtubeUrlRegex);

        let questionForVideo = ""; // Store question text specifically associated with a video

        // --- Logic Reordered: Prioritize files/URL if present, even with text ---

        if (selectedFiles.length > 0) {
            // If files are selected, process the first file
            questionForVideo = messageText; // Any text becomes the question for the file
            messageInput.value = questionForVideo; // Keep question in input (optional)

            displayMessage(`Processing file: ${selectedFiles[0].name}`, "user-message"); // Display intention
             if (questionForVideo) {
                 displayMessage(`Question (will be asked after processing): ${questionForVideo}`, "system-message");
             }

            // Create FormData and add the first file
            const formData = new FormData();
            formData.append('videoFile', selectedFiles[0]); // Backend expects 'videoFile'

            // Add the question to the form data if present
            if(questionForVideo) {
                formData.append('question', questionForVideo); // Add question if exists
            }

            processVideo(formData); // Send FormData

            // Clear selection *after* initiating upload
            selectedFiles = [];
            fileInput.value = ""; // Reset file input element
            updateFilePreview();
            messageInput.value = ""; // Clear text input too after file processing starts
            messageInput.style.height = 'auto'; // Reset textarea height

        } else if (youtubeUrlMatch) {
            // If no files, but a YouTube URL is found
            const videoUrl = youtubeUrlMatch[0];
            // Keep text that is NOT the URL as the potential question
            questionForVideo = messageText.replace(videoUrl, "").trim();
            messageInput.value = questionForVideo; // Keep question in input (optional)

            displayMessage(`Processing video URL: ${videoUrl}`, "user-message");
            if (questionForVideo) {
                displayMessage(`Question (will be asked after processing): ${questionForVideo}`, "system-message");
            }
            // Send URL and optional question as JSON
            processVideo({ videoUrl: videoUrl, question: questionForVideo });
             messageInput.value = ""; // Clear text input after URL processing starts
             messageInput.style.height = 'auto'; // Reset textarea height

        } else if (messageText) {
            // If no files or URL, just send the text as a question
            askQuestion(messageText);
            messageInput.value = ""; // Clear input after sending text question
            messageInput.style.height = 'auto'; // Reset height

        } else {
            // If nothing is entered
            displayMessage("Please enter a message, provide a Video URL, or attach a video file.", "system-message error-message");
        }
    }

    // Event listener for the send button
    sendButton.addEventListener("click", sendMessage);

    // Event listener for Enter key press in the input field (without Shift)
    messageInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault(); // Prevent default newline insertion
            sendMessage();
        }
    });

    /**
     * Sends video data (URL or File) AND optional question to the backend.
     * @param {FormData|Object} videoData - FormData containing 'videoFile' and optionally 'question',
     * OR an object containing 'videoUrl' and optionally 'question'.
     */
    function processVideo(videoData) {
        setProcessingState(true);
        displayMessage("Uploading and processing video...", "system-message"); // Inform user

        let fetchBody;
        let headers = {};
        // We need CSRF token if not using @csrf_exempt
        // headers['X-CSRFToken'] = getCookie('csrftoken'); // Add CSRF token

        if (videoData instanceof FormData) {
            fetchBody = videoData; // Already FormData for file upload
            // Don't set Content-Type for FormData, browser does it with boundary
        } else {
            // For URL, send as JSON
            fetchBody = JSON.stringify({
                videoUrl: videoData.videoUrl,
                question: videoData.question // Include question if present
            });
            headers['Content-Type'] = 'application/json';
        }

        fetch('/api/upload_video/', {
            method: 'POST',
            headers: headers,
            body: fetchBody,
        })
        .then(response => {
            if (!response.ok) {
                // Try to get error message from backend JSON response
                return response.json().then(err => {
                    throw new Error(err.error || `Server error: ${response.status}`);
                }).catch(() => {
                     // If backend didn't send JSON error
                     throw new Error(`Server error: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) { // Check for application-specific errors from backend
                 throw new Error(data.error);
            }
            // Backend now stores transcript in session/cache
            displayMessage("Video processed. Transcript ready.", "system-message"); // Use CSS for success style if desired
            // ** If a question was sent with the video, the backend might answer it directly **
            if (data.answer) {
                 displayMessage(data.answer, "assistant-message");
            } else {
                 // Or prompt user if no initial question or backend doesn't auto-answer
                 displayMessage("You can now ask questions about the video.", "system-message");
            }
        })
        .catch(error => {
            console.error("Processing Error:", error);
            displayMessage(`Error processing video: ${error.message}`, "system-message error-message");
        })
        .finally(() => {
            setProcessingState(false); // Re-enable input
        });
    }


    // Add these variables in the scope accessible by askQuestion for cahr by char printing
    let isTyping = false; // Flag to check if typing effect is active
    let chunkQueue = []; // Stores characters waiting to be typed
    const TYPING_SPEED_MS = 30; // Milliseconds between characters (adjust for speed)
    let typingIntervalId = null; // To store the interval ID
    let assistantMessageBubble = null; // Reference to the current bubble
    let accumulatedResponse = ""; // Keep track of the full response text
    let currentScrollContainer = null; // Reference to the scroll container
    let contentPlaceholder = null; // Reference to the element showing text

    /**
 * Sends a text question to the backend via POST and displays the full response.
 */
function askQuestion(question) {
    setProcessingState(true); // Disable input
    displayMessage(`${question}`, "user-message"); // Show user's question immediately

    // Use fetch to POST the question
    fetch('/api/ask_question/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            // 'X-CSRFToken': getCookie('csrftoken'), // Add CSRF token if needed
        },
        body: JSON.stringify({ question: question }), // Send question in JSON body
    })
    .then(response => {
        // Check if the response is successful
        if (!response.ok) {
            // Try to parse error JSON from backend, otherwise use status text
            return response.json().then(err => {
                // Use the error message from backend response if available
                throw new Error(err.error || `Server error: ${response.statusText}`);
            }).catch(() => {
                 // If parsing JSON error fails, just use status text
                 throw new Error(`Server error: ${response.status} ${response.statusText}`);
            });
        }
        // Parse successful response as JSON
        return response.json();
    })
    .then(data => {
        // Check for application-level errors from the backend JSON
        if (data.error) {
            throw new Error(data.error);
        }
        // Display the complete answer using the existing displayMessage function
        if (data.answer) {
            displayMessage(data.answer, "assistant-message");
        } else {
            // Handle cases where backend might not return an answer field
             displayMessage("Received an empty response.", "system-message error-message");
        }
    })
    .catch(error => {
        // Handle any errors during fetch or processing
        console.error("Asking Question Error:", error);
        // Display the error message to the user
        displayMessage(`Error: ${error.message}`, "system-message error-message");
    })
    .finally(() => {
        // Re-enable input regardless of success or failure
        setProcessingState(false);
        console.log("Ask question fetch finished.");
    });
}


    // ========================
    // Chat Sharing Function (Example - adjusted for new message structure)
    // ========================
    function shareChat() {
        let chatText = "";
        // Get messages in correct DOM order (top to bottom)
        const messages = chatMessages.querySelectorAll(".user-message, .assistant-message, .system-message, .error-message");
        messages.forEach(msg => {
            let role = "System"; // Default
            if (msg.classList.contains("user-message")) role = "User";
            else if (msg.classList.contains("assistant-message")) role = "Assistant";

            const text = msg.querySelector("p")?.textContent || msg.textContent || ""; // Get text content

            // Recreate approximate time if needed (not stored now) or skip
            // const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            // chatText += `[${time}] ${role}: ${text}\n`;
            chatText += `${role}: ${text}\n`; // Simpler format without time
        });


        if (!chatText.trim()) {
            alert("No chat messages to share.");
            return;
        }

        navigator.clipboard.writeText(chatText.trim())
            .then(() => alert("Chat copied to clipboard!"))
            .catch(err => {
                console.error("Failed to copy chat:", err);
                alert("Failed to copy chat.");
            });
    }

    // --- CSRF Token Helper (Uncomment and use if needed) ---
    // function getCookie(name) {
    //     let cookieValue = null;
    //     if (document.cookie && document.cookie !== '') {
    //         const cookies = document.cookie.split(';');
    //         for (let i = 0; i < cookies.length; i++) {
    //             const cookie = cookies[i].trim();
    //             // Does this cookie string begin with the name we want?
    //             if (cookie.substring(0, name.length + 1) === (name + '=')) {
    //                 cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
    //                 break;
    //             }
    //         }
    //     }
    //     return cookieValue;
    // }
    // --- End CSRF Token Helper ---

});