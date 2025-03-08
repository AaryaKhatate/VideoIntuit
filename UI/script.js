document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("messageInput");
    const inputFocused = localStorage.getItem("inputFocused");
    const sendButton = document.getElementById("sendBtn");
    const attachButton = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const filePreviewContainer = document.getElementById("filePreviewContainer");
    const chatMessages = document.getElementById("chatMessages");
    const plusButton = document.querySelector(".circle-btn");

    let selectedFiles = [];

    // File type icons
    const fileIcons = {
        "application/pdf": "📄", // PDF file
        "application/msword": "📝", // Word document
        "text/plain": "📃", // Text file
        "application/vnd.ms-excel": "📊", // Excel file
        "application/zip": "📦", // ZIP file
        "video/mp4": "🎥", // MP4 Video
        "video/avi": "🎥", // AVI Video
        "video/mkv": "🎥", // MKV Video
        "video/webm": "🎥", // WEBM Video
        "default": "📁" // Default file icon
    };

    // Set max-height for user messages 
    const messageElements = document.querySelectorAll(".user-message");
    messageElements.forEach(function (messageElement) {
        messageElement.style.maxHeight = window.innerHeight * 0.3 + "px";
    });

    //auto focus on first load
    console.log("inputFocused:", inputFocused);

    if (!inputFocused) {
        console.log("Auto-focusing input");
        inputField.focus();
        localStorage.setItem("inputFocused", "true");
    }

    // Open file picker when "+" is clicked
    if (plusButton) {
        plusButton.addEventListener("click", function () {
            fileInput.click();
        });
    }

    // Function to update file previews ABOVE the message bar
    function updateFilePreview() {
        filePreviewContainer.innerHTML = ""; // Clear previous preview

        selectedFiles.forEach((file, index) => {
            console.log("Processing file:", file.name, file.type); // Add console log
            const fileTag = document.createElement("div");
            fileTag.classList.add("file-tag");

            if (file.type.startsWith("image/")) {
                const imgPreview = document.createElement("img");
                imgPreview.src = URL.createObjectURL(file);
                imgPreview.classList.add("file-preview-image");
                fileTag.appendChild(imgPreview);
            } else {
                const fileIcon = document.createElement("span");
                fileIcon.textContent = fileIcons[file.type] || fileIcons["default"];
                const fileName = document.createElement("span");
                fileName.textContent = file.name;
                fileTag.appendChild(fileIcon);
                fileTag.appendChild(fileName);
            }

            const removeButton = document.createElement("button");
            removeButton.classList.add("remove-file");
            removeButton.innerHTML = "&times;";
            removeButton.onclick = function () {
                selectedFiles.splice(index, 1);
                updateFilePreview();
            };

            fileTag.appendChild(removeButton);
            filePreviewContainer.appendChild(fileTag);
        });
    }

    // Handle file selection
    fileInput.addEventListener("change", function () {
        selectedFiles = [...selectedFiles, ...fileInput.files];
        console.log("Files selected:", selectedFiles); // Add console log
        updateFilePreview();
    });

    // Function to send message with selected files
    function sendMessage() {
        const messageText = inputField.value.trim();
        if (messageText === "" && selectedFiles.length === 0) return;

        const messageBubble = document.createElement("div");
        messageBubble.classList.add("user-message");

        // Add files first
        if (selectedFiles.length > 0) {
            const fileContainer = document.createElement("div");
            fileContainer.classList.add("file-container");

            selectedFiles.forEach(file => {
                const fileBox = document.createElement("div");
                fileBox.classList.add("file-box");

                if (file.type.startsWith("image/")) {
                    const imgPreview = document.createElement("img");
                    imgPreview.src = URL.createObjectURL(file);
                    imgPreview.classList.add("file-preview-image");
                    fileBox.appendChild(imgPreview);
                } else if (file.type.startsWith("video/")) {
                    const videoIcon = document.createElement("span");
                    videoIcon.textContent = "🎥"; // Video icon

                    const fileName = document.createElement("span");
                    fileName.textContent = file.name;

                    fileBox.appendChild(videoIcon);
                    fileBox.appendChild(fileName);
                } else {
                    const fileIcon = document.createElement("span");
                    fileIcon.textContent = fileIcons[file.type] || fileIcons["default"];
                    
                    const fileName = document.createElement("span");
                    fileName.textContent = file.name;
                    
                    fileBox.appendChild(fileIcon);
                    fileBox.appendChild(fileName);
                }

                fileContainer.appendChild(fileBox);
            });

            messageBubble.appendChild(fileContainer);
        }

        // Add message text below the files
        if (messageText) {
            const textNode = document.createElement("p");
            textNode.textContent = messageText;
            textNode.classList.add("message-text");
            messageBubble.appendChild(textNode);
        }

        chatMessages.appendChild(messageBubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        inputField.value = "";
        selectedFiles = [];
        updateFilePreview();
    }

    // Send message when clicking send button
    sendButton.addEventListener("click", sendMessage);

    // Send message on Enter key press, or add new line on Shift + Enter
    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault(); // Prevent default Enter behavior (newline)
            sendMessage();
        }
    });

    // Send message on Enter key press
    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // Forcefully adjust input height
    function adjustInputHeight() {
        console.log("Adjusting height forcefully");
        inputField.style.height = "1px"; // Set to a very small height first
        inputField.style.height = (inputField.scrollHeight) + "px";
        console.log("Scroll height:", inputField.scrollHeight, "Height:", inputField.style.height);
    }

    inputField.addEventListener("input", adjustInputHeight);

    // Initial height adjustment
    adjustInputHeight();

    // Force height adjustment on every keyup
    inputField.addEventListener("keyup", adjustInputHeight);


    // Trigger height adjustment on paste events as well
    inputField.addEventListener('paste', function() {
        setTimeout(adjustInputHeight, 0); // Delay to allow paste to complete
    });

    // Trigger height adjustment on initial load and resize
    adjustInputHeight();
    window.addEventListener('resize', adjustInputHeight);
    });

    // Open file picker when "+" is clicked
    attachButton.addEventListener("click", function () {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener("change", function () {
        selectedFiles = [...selectedFiles, ...fileInput.files]; // Allow multiple files
        updateFilePreview(); // Show files ABOVE message bar
    });

    // Prevent drag behavior for watermark
    const logoWatermark = document.getElementById("logo-container");
    if (logoWatermark) {
        logoWatermark.addEventListener("dragstart", function (e) {
            e.preventDefault();
        });
    }

    // Prevent drag behavior for sign-in logo
    const signinLogo = document.getElementById("signin-button");
    if (signinLogo) {
        signinLogo.addEventListener("dragstart", function (e) {
            e.preventDefault();
        });
    }

    //sliding side bar
    document.addEventListener("DOMContentLoaded", function () {
        const sidebar = document.querySelector(".sidebar");
        const resizer = document.querySelector(".sidebar-resizer");
        let isResizing = false;
        let startX;
        let startWidth;

        resizer.addEventListener("mousedown", function (e) {
            console.log("mousedown");
            isResizing = true;
            startX = e.pageX;
            startWidth = parseInt(document.defaultView.getComputedStyle(sidebar).width, 10);
            console.log("startX:", startX, "startWidth:", startWidth);
            document.addEventListener("mousemove", resize);
            document.addEventListener("mouseup", stopResize);
        });

        function resize(e) {
            if (!isResizing) return;
            const newWidth = startWidth + e.pageX - startX;
            console.log("newWidth:", newWidth);
            sidebar.style.width = newWidth + "px";
        }

        function stopResize() {
            console.log("mouseup");
            isResizing = false;
            document.removeEventListener("mousemove", resize);
            document.removeEventListener("mouseup", stopResize);
        }
    });
