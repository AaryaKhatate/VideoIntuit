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

        inputContainer.style.minHeight = selectedFiles.length > 0 ? "160px" : "120px"; // Adjust container height

        selectedFiles.forEach(file => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");
            fileTag.textContent = file.name;
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
        if (!messageText && selectedFiles.length === 0) return;

        const messageBubble = document.createElement("div");
        messageBubble.classList.add("user-message");

        // Display selected files in the chat
        if (selectedFiles.length > 0) {
            const fileContainer = document.createElement("div");
            fileContainer.classList.add("file-container");

            selectedFiles.forEach(file => {
                const fileBox = document.createElement("div");
                fileBox.classList.add("file-box");
                fileBox.textContent = file.name;
                fileContainer.appendChild(fileBox);
            });

            messageBubble.appendChild(fileContainer);
        }

        // Display message text in the chat
        if (messageText) {
            const textNode = document.createElement("p");
            textNode.classList.add("message-text");
            textNode.textContent = messageText;
            messageBubble.appendChild(textNode);
        }

        // Add message to the top of chat
        chatMessages.prepend(messageBubble);

        // Reset input field and selected files
        inputField.value = "";
        selectedFiles = [];
        updateFilePreview();
    }

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

    // ========================
    // Event Listeners
    // ========================

    sendButton.addEventListener("click", sendMessage);

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
});
