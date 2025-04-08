document.addEventListener("DOMContentLoaded", function () {
    // ========================
    // DOM Elements
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
    // File Handling
    // ========================

    function updateFilePreview() {
        filePreviewContainer.innerHTML = "";

        if (selectedFiles.length > 0) {
            filePreviewContainer.style.display = "flex";
            inputContainer.classList.add("expanded");

            selectedFiles.forEach((file, index) => {
                const fileTag = document.createElement("div");
                fileTag.classList.add("file-tag");

                const icon = document.createElement("i");
                icon.classList.add("fas", "fa-video", "file-icon");

                const fileName = document.createElement("span");
                fileName.textContent = file.name;

                const closeBtn = document.createElement("span");
                closeBtn.classList.add("file-close");
                closeBtn.innerHTML = "&times;";
                closeBtn.title = "Remove file";

                closeBtn.addEventListener("click", () => {
                    selectedFiles.splice(index, 1);
                    updateFilePreview();
                    fileInput.value = "";
                });

                fileTag.appendChild(icon);
                fileTag.appendChild(fileName);
                fileTag.appendChild(closeBtn);

                filePreviewContainer.appendChild(fileTag);
            });
        } else {
            filePreviewContainer.style.display = "none";
            inputContainer.classList.remove("expanded");
        }
    }

    function autoResizeInput() {
        inputField.style.height = "45px";
        const scrollHeight = inputField.scrollHeight;
        const maxHeight = 1.5 * 16 * 6 + 20;

        if (scrollHeight <= maxHeight) {
            inputField.style.overflowY = "hidden";
            inputField.style.height = scrollHeight + "px";
        } else {
            inputField.style.overflowY = "auto";
            inputField.style.height = maxHeight + "px";
        }
    }

    fileInput.addEventListener("change", function () {
        const validVideos = [...fileInput.files].filter(file => file.type.startsWith("video/"));

        for (const file of validVideos) {
            const alreadyAdded = selectedFiles.some(f => f.name === file.name && f.size === file.size);
            if (!alreadyAdded) {
                selectedFiles.push(file);
            }
        }

        updateFilePreview();
        fileInput.value = "";
    });

    attachButton.addEventListener("click", () => fileInput.click());

    // ========================
    // Chat Messaging
    // ========================

    function sendMessage() {
        const messageText = inputField.value.trim();
        if (!messageText && selectedFiles.length === 0) return;

        const messageBubble = document.createElement("div");
        messageBubble.classList.add("user-message");

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

        if (messageText) {
            const textNode = document.createElement("p");
            textNode.classList.add("message-text");
            textNode.textContent = messageText;
            messageBubble.appendChild(textNode);
        }

        chatMessages.prepend(messageBubble);

        inputField.value = "";
        selectedFiles = [];
        updateFilePreview();
        autoResizeInput();
    }

    // ========================
    // Chat Sharing
    // ========================

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

    inputField.addEventListener("input", autoResizeInput);

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
