// chatbot.js

let chatHistory = [];
let currentApiKey = ''; // Store the API key
let selectedModel = 'gemini-2.5-flash-lite'; // Default model changed to gemini-2.5-flash-lite

let totalInputTokens = 0;
let totalOutputTokens = 0;

// Get DOM elements
const geminiApiKeyInput = document.getElementById('geminiApiKey');
const setApiKeyButton = document.getElementById('setApiKeyButton');
const geminiModelSelect = document.getElementById('geminiModel'); // New element
const chatHistoryDiv = document.getElementById('chatHistory');
const messageInput = document.getElementById('messageInput');
const sendMessageButton = document.getElementById('sendMessageButton');
const errorMessageDiv = document.getElementById('errorMessage');
const tokenStatsDiv = document.getElementById('tokenStats'); // New element for token stats

// New DOM elements for raw chat history
const rawChatHistoryInput = document.getElementById('rawChatHistoryInput');
const applyRawHistoryButton = document.getElementById('applyRawHistoryButton');


// Utility functions for cookies
function setCookie(name, value, days) {
    const d = new Date();
    d.setTime(d.getTime() + (days * 24 * 60 * 60 * 1000));
    const expires = "expires=" + d.toUTCString();
    document.cookie = name + "=" + encodeURIComponent(value) + ";" + expires + ";path=/;SameSite=Lax";
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for(let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) return decodeURIComponent(c.substring(nameEQ.length, c.length));
    }
    return null;
}

// Function to validate and store the API key
function setApiKey() {
    const apiKey = geminiApiKeyInput.value.trim();
    if (!apiKey) {
        errorMessageDiv.textContent = 'Please enter your Gemini API Key.';
        currentApiKey = '';
        return false;
    }
    currentApiKey = apiKey;
    setCookie('geminiApiKey', apiKey, 30); // Save API key to cookie for 30 days
    errorMessageDiv.textContent = 'API Key set successfully and saved!'; // Indicate success
    setTimeout(() => errorMessageDiv.textContent = '', 3000); // Clear after 3 seconds
    console.log('API Key set.');
    return true;
}

// Function to load the API key from cookie
function loadApiKeyFromCookie() {
    const apiKey = getCookie('geminiApiKey');
    if (apiKey) {
        geminiApiKeyInput.value = apiKey;
        currentApiKey = apiKey;
        errorMessageDiv.textContent = 'API Key loaded from cookie!';
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        console.log('API Key loaded from cookie.');
    }
}

// Function to update the selected model
function updateSelectedModel() {
    selectedModel = geminiModelSelect.value;
    console.log(`Selected model: ${selectedModel}`);
    errorMessageDiv.textContent = `Model set to: ${selectedModel}`;
    setTimeout(() => errorMessageDiv.textContent = '', 3000); // Clear after 3 seconds
}

// Function to update the raw chat history textarea
function updateRawHistoryInput() {
    if (rawChatHistoryInput) {
        // Exclude the initial welcome message from the editable raw history,
        // as it's typically a UI-only element or the first model message.
        // For simplicity, we'll include all messages in chatHistory for now.
        // If the initial welcome message from HTML is pushed, it's part of chatHistory.
        try {
            rawChatHistoryInput.value = JSON.stringify(chatHistory, null, 2); // Pretty print JSON
        } catch (e) {
            console.error("Error stringifying chat history:", e);
            rawChatHistoryInput.value = "Error: Could not display chat history as JSON.";
        }
    }
}

// Function to apply raw chat history from the textarea
function applyRawHistory() {
    if (!rawChatHistoryInput) return;

    const rawText = rawChatHistoryInput.value;
    try {
        const parsedHistory = JSON.parse(rawText);
        // Basic validation: ensure it's an array and each item has 'role' and 'parts'
        if (!Array.isArray(parsedHistory) || !parsedHistory.every(item => item.role && Array.isArray(item.parts))) {
            throw new Error("Invalid chat history format. Expected an array of objects with 'role' and 'parts'.");
        }
        chatHistory = parsedHistory;
        renderChatHistory(); // Re-render chat bubbles based on new history
        errorMessageDiv.textContent = 'Chat history applied successfully!';
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        console.log('Chat history updated from raw input.');
    } catch (error) {
        console.error('Error applying raw chat history:', error);
        errorMessageDiv.textContent = `Error applying chat history: ${error.message}`;
    }
}


// Function to render chat history to the UI
function renderChatHistory() {
    chatHistoryDiv.innerHTML = ''; // Clear existing messages
    chatHistory.forEach((msg) => {
        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble');
        messageBubble.classList.add(msg.role === 'user' ? 'user-message' : 'model-message');
        
        const paragraph = document.createElement('p');
        paragraph.textContent = msg.parts[0].text;
        messageBubble.appendChild(paragraph);

        chatHistoryDiv.appendChild(messageBubble);
    });
    // Scroll to the bottom
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;

    // Update the raw history input whenever the visual chat history is rendered
    updateRawHistoryInput();
}

// Function to render accumulated token stats
function renderTokenStats() {
    if (tokenStatsDiv) {
        tokenStatsDiv.textContent = `Input Tokens: ${totalInputTokens} | Output Tokens: ${totalOutputTokens}`;
    }
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
    renderChatHistory(); // Render the new user message and update raw history input
    messageInput.value = ''; // Clear input
    adjustTextareaHeight(); // Reset textarea height
    errorMessageDiv.textContent = 'Thinking...'; // Show thinking indicator

    try {
        const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`;

        // The API expects `contents` to be the full chat history up to this point
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

        // Update token counts
        if (data.usageMetadata) {
            totalInputTokens += data.usageMetadata.promptTokenCount || 0;
            totalOutputTokens += data.usageMetadata.candidatesTokenCount || 0;
            renderTokenStats();
        }

        // Add model response to history
        chatHistory.push({ role: 'model', parts: [{ text: modelResponseText }] });
        errorMessageDiv.textContent = ''; // Clear thinking message
        renderChatHistory(); // Render the new model message and update raw history input

    } catch (error) {
        console.error('Error sending message:', error);
        errorMessageDiv.textContent = `Error sending message: ${error.message}`;
        // If API call fails, remove the last user message from history
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            chatHistory.pop();
            renderChatHistory(); // Re-render to reflect removal and update raw history input
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
applyRawHistoryButton.addEventListener('click', applyRawHistory); // Event listener for applying raw history

messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent new line
        sendMessage();
    }
});

messageInput.addEventListener('input', adjustTextareaHeight);

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    loadApiKeyFromCookie(); // Load API key from cookie
    
    // Add the initial welcome message from HTML to the chatHistory array if it's not already there
    // This ensures it's part of the history when sending to the API.
    const initialWelcomeMessageElement = chatHistoryDiv.querySelector('.message-bubble.model-message p');
    if (initialWelcomeMessageElement) {
        const welcomeText = initialWelcomeMessageElement.textContent.trim();
        // Check if chatHistory is empty or if the first message is not the welcome text
        if (chatHistory.length === 0 || (chatHistory.length > 0 && chatHistory[0].parts[0].text !== welcomeText)) {
             chatHistory.unshift({ role: 'model', parts: [{ text: welcomeText }] }); // Add to the beginning
        }
    }
    renderChatHistory(); // Render the initial history (including welcome message) and update raw input
    adjustTextareaHeight(); // Adjust textarea height on page load

    // Set the initial selected model based on dropdown and update global variable
    geminiModelSelect.value = selectedModel; // Ensure the dropdown reflects the default
    updateSelectedModel(); 
    renderTokenStats(); // Render initial token stats (should be 0)
});