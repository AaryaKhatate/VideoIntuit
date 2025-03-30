document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const filePreviewContainer = document.getElementById("filePreviewContainer");
    const chatMessages = document.getElementById("chatMessages");
    const signinButton = document.getElementById("signin-button");
    const shareButton = document.querySelector(".share-btn");
 
 
    let selectedFiles = [];
 
 
    /**
     * Function to send a message
     */
    function sendMessage() {
        const messageText = inputField.value.trim();
        if (!messageText && selectedFiles.length === 0) return;
 
 
        const messageBubble = document.createElement("div");
        messageBubble.classList.add("user-message");
 
 
        // Display selected files
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
 
 
        // Display message text
        if (messageText) {
            const textNode = document.createElement("p");
            textNode.classList.add("message-text");
            textNode.textContent = messageText;
            messageBubble.appendChild(textNode);
        }
 
 
        // Add message at the top
        chatMessages.prepend(messageBubble);
 
 
        // Reset input field and selected files
        inputField.value = "";
        selectedFiles = [];
        updateFilePreview();
    }
 
 
    /**
     * Function to update file preview
     */
    function updateFilePreview() {
        filePreviewContainer.innerHTML = "";
        selectedFiles.forEach(file => {
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");
            fileTag.textContent = file.name;
            filePreviewContainer.appendChild(fileTag);
        });
    }
 
 
    /**
     * Function to handle chat sharing
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
 
 
    // Event Listeners
    sendButton.addEventListener("click", sendMessage);
 
 
    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
 
 
    fileInput.addEventListener("change", function () {
        selectedFiles = [...selectedFiles, ...fileInput.files];
        updateFilePreview();
    });
 
 
    if (shareButton) {
        shareButton.addEventListener("click", shareChat);
    }
 
 
    signinButton.addEventListener("click", function () {
        alert("Sign-in button clicked! Implement authentication logic.");
    });
 });
 
 
 