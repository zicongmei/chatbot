// chatbot.js

let chatHistory = [];
let currentApiKey = ''; // Store the API key
let selectedModel = 'gemini-2.5-pro'; // Default model

// Get DOM elements
const geminiApiKeyInput = document.getElementById('geminiApiKey');
const setApiKeyButton = document.getElementById('setApiKeyButton');
const geminiModelSelect = document.getElementById('geminiModel'); // New element
const chatHistoryDiv = document.getElementById('chatHistory');
const messageInput = document.getElementById('messageInput');
const sendMessageButton = document.getElementById('sendMessageButton');
const errorMessageDiv = document.getElementById('errorMessage');

// Function to validate and store the API key
function setApiKey() {
    const apiKey = geminiApiKeyInput.value.trim();
    if (!apiKey) {
        errorMessageDiv.textContent = 'Please enter your Gemini API Key.';
        currentApiKey = '';
        return false;
    }
    currentApiKey = apiKey;
    errorMessageDiv.textContent = 'API Key set successfully!'; // Indicate success
    setTimeout(() => errorMessageDiv.textContent = '', 3000); // Clear after 3 seconds
    console.log('API Key set.');
    return true;
}

// Function to update the selected model
function updateSelectedModel() {
    selectedModel = geminiModelSelect.value;
    console.log(`Selected model: ${selectedModel}`);
    errorMessageDiv.textContent = `Model set to: ${selectedModel}`;
    setTimeout(() => errorMessageDiv.textContent = '', 3000); // Clear after 3 seconds
}

// Function to render chat history to the UI
function renderChatHistory() {
    chatHistoryDiv.innerHTML = ''; // Clear existing messages
    chatHistory.forEach((msg, index) => {
        // Only render actual user/model messages, not the initial welcome text placeholder
        if (msg.role === 'user' || msg.role === 'model') {
            const messageBubble = document.createElement('div');
            messageBubble.classList.add('message-bubble');
            messageBubble.classList.add(msg.role === 'user' ? 'user-message' : 'model-message');
            messageBubble.setAttribute('data-index', index); // Store index for editing
            
            // Making messages contenteditable for editing chat history
            // Only user messages can be directly edited, model messages are read-only
            if (msg.role === 'user') {
                 messageBubble.setAttribute('contenteditable', 'true'); 
            }
           
            const paragraph = document.createElement('p');
            paragraph.textContent = msg.parts[0].text;
            messageBubble.appendChild(paragraph);

            // Add blur event listener for editing
            if (msg.role === 'user') {
                messageBubble.addEventListener('blur', async (event) => {
                    const newText = event.target.textContent.trim();
                    const idx = parseInt(event.target.getAttribute('data-index'));
                    if (chatHistory[idx] && chatHistory[idx].parts[0].text !== newText) {
                        chatHistory[idx].parts[0].text = newText;
                        console.log(`Message at index ${idx} updated: "${newText}"`);
                        // The updated history will be sent with the next message.
                    }
                });

                // Prevent adding new line on enter in contenteditable, instead blur
                messageBubble.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        event.target.blur(); // Trigger blur to save changes
                    }
                });
            }

            chatHistoryDiv.appendChild(messageBubble);
        }
    });
    // Scroll to the bottom
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

// Function to send a message directly via HTTP request to Gemini endpoint
async function sendMessage() {
    const userMessageText = messageInput.value.trim();
    if (!userMessageText) {
        return; // Don't send empty messages
    }

    if (!currentApiKey) {
        errorMessageDiv.textContent = 'Please set your Gemini API Key first.';
        return;
    }

    // Add user message to history
    chatHistory.push({ role: 'user', parts: [{ text: userMessageText }] });
    renderChatHistory();
    messageInput.value = ''; // Clear input
    adjustTextareaHeight(); // Reset textarea height
    errorMessageDiv.textContent = 'Thinking...'; // Show thinking indicator

    try {
        const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`;

        // The API expects `contents` to be the full chat history up to this point
        // The current `chatHistory` structure ({ role: 'user/model', parts: [{ text: "..." }] })
        // is compatible with the API's expected format.
        const requestBody = {
            contents: chatHistory,
            generationConfig: {
                maxOutputTokens: 5000,
            },
        };

        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': currentApiKey,
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json();
            const errorMessage = errorData.error ? errorData.error.message : response.statusText;
            throw new Error(`API Error: ${errorMessage} (Status: ${response.status})`);
        }

        const data = await response.json();
        
        // Extract model's response text from the expected structure
        const modelResponseText = data.candidates && data.candidates.length > 0 &&
                                  data.candidates[0].content && data.candidates[0].content.parts &&
                                  data.candidates[0].content.parts.length > 0
                                  ? data.candidates[0].content.parts[0].text
                                  : 'No response from model.';

        // Add model response to history
        chatHistory.push({ role: 'model', parts: [{ text: modelResponseText }] });
        errorMessageDiv.textContent = ''; // Clear thinking message
        renderChatHistory();

    } catch (error) {
        console.error('Error sending message:', error);
        errorMessageDiv.textContent = `Error sending message: ${error.message}`;
        // If API call fails, remove the last user message from history
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            chatHistory.pop();
            renderChatHistory(); // Re-render to reflect removal
        }
    }
}

// Adjust textarea height based on content
function adjustTextareaHeight() {
    messageInput.style.height = 'auto'; // Reset height to recalculate
    messageInput.style.height = messageInput.scrollHeight + 'px';
    if (messageInput.scrollHeight > 150) { // Max height limit for textarea
        messageInput.style.height = '150px';
        messageInput.style.overflowY = 'auto';
    } else {
        messageInput.style.overflowY = 'hidden';
    }
}

// Event Listeners
setApiKeyButton.addEventListener('click', setApiKey);
geminiModelSelect.addEventListener('change', updateSelectedModel); // Event listener for model selection
sendMessageButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent new line
        sendMessage();
    }
});

messageInput.addEventListener('input', adjustTextareaHeight);

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    // Add the initial welcome message from HTML to the chatHistory array
    const initialWelcomeMessageElement = chatHistoryDiv.querySelector('.message-bubble.model-message p');
    if (initialWelcomeMessageElement) {
        const welcomeText = initialWelcomeMessageElement.textContent.trim();
        if (welcomeText && chatHistory.length === 0) {
             chatHistory.push({ role: 'model', parts: [{ text: welcomeText }] });
        }
    }
    renderChatHistory(); // Render the initial history (including welcome message)
    adjustTextareaHeight(); // Adjust textarea height on page load
    updateSelectedModel(); // Set the initial selected model based on dropdown
});