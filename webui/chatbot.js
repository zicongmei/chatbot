// chatbot.js

let chatHistory = [];
let currentApiKey = '';
let selectedModel = 'gemini-2.5-flash-lite';
let systemInstruction = ''; // New variable for system instruction

let totalInputTokens = 0;
let totalOutputTokens = 0;

// Get DOM elements
const geminiApiKeyInput = document.getElementById('geminiApiKey');
const setApiKeyButton = document.getElementById('setApiKeyButton');
const geminiModelSelect = document.getElementById('geminiModel');
const chatHistoryDiv = document.getElementById('chatHistory');
const messageInput = document.getElementById('messageInput');
const sendMessageButton = document.getElementById('sendMessageButton');
const errorMessageDiv = document.getElementById('errorMessage');
const tokenStatsDiv = document.getElementById('tokenStats');

// DOM elements for raw chat history
const rawChatHistoryInput = document.getElementById('rawChatHistoryInput');
const applyRawHistoryButton = document.getElementById('applyRawHistoryButton');

// New DOM elements for system instruction
const systemInstructionInput = document.getElementById('systemInstructionInput');

// New DOM elements for chat save/load
const saveChatButton = document.getElementById('saveChatButton');
const loadChatButton = document.getElementById('loadChatButton');
const loadChatFileInput = document.getElementById('loadChatFileInput');


// Utility functions for localStorage
function setLocalStorageItem(name, value) {
    try {
        localStorage.setItem(name, value);
    } catch (e) {
        console.error(`Error saving to localStorage for ${name}:`, e);
        errorMessageDiv.textContent = `Error saving data locally: ${e.message}`;
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
    }
}

function getLocalStorageItem(name) {
    try {
        return localStorage.getItem(name);
    } catch (e) {
        console.error(`Error loading from localStorage for ${name}:`, e);
        errorMessageDiv.textContent = `Error loading data locally: ${e.message}`;
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return null;
    }
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
    setLocalStorageItem('geminiApiKey', apiKey); // Save API key to localStorage
    errorMessageDiv.textContent = 'API Key set successfully and saved!';
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
    console.log('API Key set.');
    return true;
}

// Function to load the API key from localStorage
function loadApiKeyFromLocalStorage() {
    const apiKey = getLocalStorageItem('geminiApiKey');
    if (apiKey) {
        geminiApiKeyInput.value = apiKey;
        currentApiKey = apiKey;
        errorMessageDiv.textContent = 'API Key loaded from local storage!';
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        console.log('API Key loaded from local storage.');
    }
}

// Function to update the selected model
function updateSelectedModel() {
    selectedModel = geminiModelSelect.value;
    console.log(`Selected model: ${selectedModel}`);
    errorMessageDiv.textContent = `Model set to: ${selectedModel}`;
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
}

// Function to save system instruction to localStorage
function saveSystemInstruction() {
    systemInstruction = systemInstructionInput.value.trim();
    setLocalStorageItem('systemInstruction', systemInstruction);
    console.log('System instruction saved.');
}

// Function to load system instruction from localStorage
function loadSystemInstructionFromLocalStorage() {
    const loadedInstruction = getLocalStorageItem('systemInstruction');
    if (loadedInstruction) {
        systemInstruction = loadedInstruction;
        systemInstructionInput.value = loadedInstruction;
        console.log('System instruction loaded from local storage.');
    }
}

// Function to save chat history to localStorage
function saveChatHistoryToLocalStorage() {
    setLocalStorageItem('chatHistory', JSON.stringify(chatHistory));
    console.log('Chat history saved to local storage.');
}

// Function to load chat history from localStorage
function loadChatHistoryFromLocalStorage() {
    const storedChatHistory = getLocalStorageItem('chatHistory');
    if (storedChatHistory) {
        try {
            const parsedHistory = JSON.parse(storedChatHistory);
            if (Array.isArray(parsedHistory)) {
                chatHistory = parsedHistory;
                console.log('Chat history loaded from local storage.');
            } else {
                console.warn('Stored chat history is not an array, initializing empty.');
                chatHistory = [];
            }
        } catch (e) {
            console.error('Error parsing stored chat history from local storage:', e);
            chatHistory = []; // Reset on error
        }
    }
    // If no history in localStorage or parsing error, initialize with welcome message
    if (chatHistory.length === 0) {
        const initialWelcomeMessageElement = chatHistoryDiv.querySelector('.welcome-message p');
        if (initialWelcomeMessageElement) {
            const welcomeText = initialWelcomeMessageElement.textContent.trim();
            chatHistory.push({ role: 'model', parts: [{ text: welcomeText }] });
        }
    }
}

// Function to update the raw chat history textarea
function updateRawHistoryInput() {
    if (rawChatHistoryInput) {
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
        if (!Array.isArray(parsedHistory) || !parsedHistory.every(item => item.role && Array.isArray(item.parts))) {
            throw new Error("Invalid chat history format. Expected an array of objects with 'role' and 'parts'.");
        }
        chatHistory = parsedHistory;
        renderChatHistory(); // Re-render chat bubbles based on new history
        saveChatHistoryToLocalStorage(); // Save updated history
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
    saveChatHistoryToLocalStorage(); // Save updated history
    messageInput.value = ''; // Clear input
    adjustTextareaHeight(); // Reset textarea height
    errorMessageDiv.textContent = 'Thinking...'; // Show thinking indicator

    try {
        const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`;

        // Prepend system instruction if available
        const conversationContent = [...chatHistory];
        if (systemInstruction) {
            // For Gemini models, system instructions are typically handled by inserting an initial user message.
            // Some models might interpret a user message as system instruction if it's the very first.
            // A common pattern is user: [system instruction] -> model: [empty/acknowledgement]
            // For simplicity, we'll just prepend it as a user message.
            conversationContent.unshift({ role: 'user', parts: [{ text: systemInstruction }] });
            // If the model expects a paired empty model response to set the context, you might add:
            // conversationContent.unshift({ role: 'model', parts: [{ text: '' }] });
        }


        const requestBody = {
            contents: conversationContent,
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
        saveChatHistoryToLocalStorage(); // Save updated history

    } catch (error) {
        console.error('Error sending message:', error);
        errorMessageDiv.textContent = `Error sending message: ${error.message}`;
        // If API call fails, remove the last user message from history
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            chatHistory.pop();
            renderChatHistory(); // Re-render to reflect removal and update raw history input
            saveChatHistoryToLocalStorage(); // Save updated history
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

// Function to download chat history as a JSON file
function downloadChatHistory() {
    if (chatHistory.length === 0) {
        errorMessageDiv.textContent = "No chat history to save.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }
    const filename = `gemini_chat_history_${new Date().toISOString().slice(0, 10)}.json`;
    const jsonStr = JSON.stringify(chatHistory, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    errorMessageDiv.textContent = "Chat history saved to file.";
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
}

// Function to handle loading chat history from a file
function handleChatFileLoad(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const loadedHistory = JSON.parse(e.target.result);
            if (!Array.isArray(loadedHistory) || !loadedHistory.every(item => item.role && Array.isArray(item.parts))) {
                throw new Error("Invalid chat history file format. Expected an array of objects with 'role' and 'parts'.");
            }
            chatHistory = loadedHistory;
            renderChatHistory();
            saveChatHistoryToLocalStorage(); // Save loaded history to local storage
            errorMessageDiv.textContent = "Chat history loaded from file successfully!";
        } catch (error) {
            console.error('Error loading chat history from file:', error);
            errorMessageDiv.textContent = `Error loading chat history from file: ${error.message}`;
        } finally {
            setTimeout(() => errorMessageDiv.textContent = '', 5000);
            loadChatFileInput.value = ''; // Clear the file input
        }
    };
    reader.onerror = (e) => {
        console.error('Error reading file:', e);
        errorMessageDiv.textContent = `Error reading file: ${e.target.error.message}`;
        setTimeout(() => errorMessageDiv.textContent = '', 5000);
        loadChatFileInput.value = ''; // Clear the file input
    };
    reader.readAsText(file);
}


// Event Listeners
setApiKeyButton.addEventListener('click', setApiKey);
geminiModelSelect.addEventListener('change', updateSelectedModel);
sendMessageButton.addEventListener('click', sendMessage);
applyRawHistoryButton.addEventListener('click', applyRawHistory);

// System Instruction events
systemInstructionInput.addEventListener('input', saveSystemInstruction);

// Chat Save/Load events
saveChatButton.addEventListener('click', downloadChatHistory);
loadChatButton.addEventListener('click', () => loadChatFileInput.click()); // Trigger file input click
loadChatFileInput.addEventListener('change', handleChatFileLoad);


messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent new line
        sendMessage();
    }
});
messageInput.addEventListener('input', adjustTextareaHeight);

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    loadApiKeyFromLocalStorage(); // Load API key
    loadSystemInstructionFromLocalStorage(); // Load system instruction
    loadChatHistoryFromLocalStorage(); // Load chat history (or initialize with welcome)
    
    renderChatHistory(); // Render the initial history (including welcome message) and update raw input
    adjustTextareaHeight(); // Adjust textarea height on page load

    // Set the initial selected model based on dropdown and update global variable
    geminiModelSelect.value = selectedModel; // Ensure the dropdown reflects the default
    updateSelectedModel(); 
    renderTokenStats(); // Render initial token stats (should be 0)
});