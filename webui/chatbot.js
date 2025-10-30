// chatbot.js

let chatHistory = [];
let currentApiKey = '';
let selectedModel = 'gemini-2.5-flash-lite';
let systemInstruction = ''; // New variable for system instruction

let totalInputTokens = 0;
let totalOutputTokens = 0;
let currentInputTokens = 0; // New: Tokens for the current request
let currentOutputTokens = 0; // New: Tokens for the current request
let lastRemovedWasModelReply = false; // New: To track if last removed entry was a model reply

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
const debugButton = document.getElementById('debugButton'); // New: Debug button
const rawChatContent = document.getElementById('rawChatContent'); // New: Raw chat content div

// New DOM elements for system instruction
const systemInstructionInput = document.getElementById('systemInstructionInput');
const clearSystemInstructionButton = document.getElementById('clearSystemInstructionButton'); // New: Clear system instruction button

// New DOM elements for chat save/load
const saveChatButton = document.getElementById('saveChatButton');
const loadChatButton = document.getElementById('loadChatButton');
const loadChatFileInput = document.getElementById('loadChatFileInput');

// New DOM elements for chat history actions
const removeLastEntryButton = document.getElementById('removeLastEntryButton'); // New
const clearAllHistoryButton = document.getElementById('clearAllHistoryButton'); // New
const regenerateSystemReplyButton = document.getElementById('regenerateSystemReplyButton'); // New


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
    setLocalStorageItem('selectedModel', selectedModel); // Save selected model to localStorage
    console.log(`Selected model: ${selectedModel}`);
    errorMessageDiv.textContent = `Model set to: ${selectedModel}`;
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
}

// Function to load the selected model from localStorage
function loadSelectedModelFromLocalStorage() {
    const storedModel = getLocalStorageItem('selectedModel');
    if (storedModel) {
        selectedModel = storedModel;
        geminiModelSelect.value = storedModel;
        console.log(`Selected model loaded from local storage: ${selectedModel}`);
    } else {
        // If no model is stored, ensure the dropdown reflects the default
        geminiModelSelect.value = selectedModel;
    }
}

// Function to save system instruction to localStorage
function saveSystemInstruction() {
    systemInstruction = systemInstructionInput.value.trim();
    setLocalStorageItem('systemInstruction', systemInstruction);
    console.log('System instruction saved.');
    updateRawHistoryInput(); // Update raw history display when system instruction changes
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

// Function to clear system instruction
function clearSystemInstruction() {
    if (systemInstructionInput.value.trim() === '') {
        errorMessageDiv.textContent = "Background instruction is already empty.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }
    if (confirm('Are you sure you want to clear the background / system instruction?')) {
        systemInstructionInput.value = '';
        systemInstruction = '';
        saveSystemInstruction(); // This also updates local storage and raw history
        errorMessageDiv.textContent = "Background instruction cleared.";
    }
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
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
    // If no history in localStorage or parsing error, or history is empty after loading,
    // and there's no system instruction already providing context, add a welcome message.
    if (chatHistory.length === 0 && !systemInstruction) {
        // don't show the welcome message
        // chatHistory.push({ role: 'model', parts: [{ text: 'Hello! Please enter your Gemini API key and select a model above to start chatting.' }] });
        // console.log('Initialized chat history with a welcome message.');
        saveChatHistoryToLocalStorage(); // Save this initial state
    }
}

// Function to save token stats to localStorage
function saveTokenStatsToLocalStorage() {
    setLocalStorageItem('totalInputTokens', totalInputTokens.toString());
    setLocalStorageItem('totalOutputTokens', totalOutputTokens.toString());
    console.log('Token stats saved to local storage.');
}

// Function to load token stats from localStorage
function loadTokenStatsFromLocalStorage() {
    const storedInput = getLocalStorageItem('totalInputTokens');
    const storedOutput = getLocalStorageItem('totalOutputTokens');
    if (storedInput) {
        totalInputTokens = parseInt(storedInput, 10);
    }
    if (storedOutput) {
        totalOutputTokens = parseInt(storedOutput, 10);
    }
    console.log(`Token stats loaded: Input=${totalInputTokens}, Output=${totalOutputTokens}`);
}

// Function to update the raw chat history textarea
function updateRawHistoryInput() {
    if (rawChatHistoryInput) {
        try {
            const dataToDisplay = {
                systemInstruction: systemInstruction,
                chatHistory: chatHistory
            };
            rawChatHistoryInput.value = JSON.stringify(dataToDisplay, null, 2); // Pretty print JSON
        } catch (e) {
            console.error("Error stringifying chat history for raw input:", e);
            rawChatHistoryInput.value = "Error: Could not display chat history as JSON.";
        }
    }
}

// Function to apply raw chat history from the textarea
function applyRawHistory() {
    if (!rawChatHistoryInput) return;

    chatHistory = []

    const rawText = rawChatHistoryInput.value;
    try {
        const parsedData = JSON.parse(rawText);

        if (typeof parsedData !== 'object' || parsedData === null) {
            throw new Error("Invalid JSON format. Expected an object with 'systemInstruction' and 'chatHistory'.");
        }

        // Apply system instruction
        const newSystemInstruction = parsedData.systemInstruction || '';
        if (typeof newSystemInstruction !== 'string') {
            throw new Error("Invalid 'systemInstruction' format. Expected a string.");
        }
        systemInstruction = newSystemInstruction;
        systemInstructionInput.value = systemInstruction;
        setLocalStorageItem('systemInstruction', systemInstruction); // Save to local storage

        // Apply chat history
        const newChatHistory = parsedData.chatHistory;
        if (!Array.isArray(newChatHistory) || !newChatHistory.every(item => item.role && Array.isArray(item.parts))) {
            throw new Error("Invalid 'chatHistory' format. Expected an array of objects with 'role' and 'parts'.");
        }
        chatHistory = newChatHistory;
        renderChatHistory(); // Re-render chat bubbles based on new history
        saveChatHistoryToLocalStorage(); // Save updated history to local storage

        errorMessageDiv.textContent = 'Chat history and system instruction applied successfully!';
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        console.log('Chat history and system instruction updated from raw input.');
    } catch (error) {
        console.error('Error applying raw chat history:', error);
        errorMessageDiv.textContent = `Error applying raw chat history: ${error.message}`;
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

    updateRawHistoryInput(); // Ensure raw history input is updated after rendering
}

// Function to render accumulated token stats
function renderTokenStats() {
    if (tokenStatsDiv) {
        tokenStatsDiv.innerHTML = `
            <div><strong>Input Tokens:</strong> Current: ${currentInputTokens} | Total: ${totalInputTokens}</div>
            <div><strong>Output Tokens:</strong> Current: ${currentOutputTokens} | Total: ${totalOutputTokens}</div>
        `;
    }
}

// Helper function to send content to the Gemini API
async function _sendContentToModel(userMessageTextForAPI, contentToSendForAPI) {
    if (!currentApiKey) {
        errorMessageDiv.textContent = 'Please set your Gemini API Key first.';
        return false; // Indicate failure
    }

    errorMessageDiv.textContent = 'Thinking...'; // Show thinking indicator

    // Reset current request token counts at the start of a new API call attempt
    currentInputTokens = 0;
    currentOutputTokens = 0;
    renderTokenStats(); // Update UI to reflect reset

    try {
        const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`;

        const requestBody = {
            contents: contentToSendForAPI, // This will be the actual history for the API call
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
        
        const modelResponseText = data.candidates && data.candidates.length > 0 &&
                                  data.candidates[0].content && data.candidates[0].content.parts &&
                                  data.candidates[0].content.parts.length > 0
                                  ? data.candidates[0].content.parts[0].text
                                  : 'No response from model.';

        // Update token counts
        if (data.usageMetadata) {
            currentInputTokens = data.usageMetadata.promptTokenCount || 0; // Update current request tokens
            currentOutputTokens = data.usageMetadata.candidatesTokenCount || 0; // Update current request tokens

            totalInputTokens += currentInputTokens;
            totalOutputTokens += currentOutputTokens;
            renderTokenStats();
            saveTokenStatsToLocalStorage();
        }

        // Add model response to history
        chatHistory.push({ role: 'model', parts: [{ text: modelResponseText }] });
        errorMessageDiv.textContent = ''; // Clear thinking message
        renderChatHistory(); // Render the new model message and update raw history input
        saveChatHistoryToLocalStorage(); // Save updated history
        return true; // Indicate success

    } catch (error) {
        console.error('Error sending message:', error);
        errorMessageDiv.textContent = `Error sending message: ${error.message}`;
        // On error, current tokens should be 0 as the request failed or was incomplete.
        currentInputTokens = 0;
        currentOutputTokens = 0;
        renderTokenStats(); // Update UI to reflect 0 for current
        return false; // Indicate failure
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

    // Prepare content for API call, including system instruction
    const conversationContent = [...chatHistory];
    if (systemInstruction) {
        // For Gemini models, system instructions are typically handled by inserting an initial user message.
        // This does not modify the persistent chatHistory or displayed raw history.
        conversationContent.unshift({ role: 'user', parts: [{ text: systemInstruction }] });
    }

    const success = await _sendContentToModel(userMessageText, conversationContent);

    if (!success) {
        // If API call failed, remove the last user message from history
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            chatHistory.pop();
            renderChatHistory(); // Re-render to reflect removal and update raw history input
            saveChatHistoryToLocalStorage(); // Save updated history
        }
    }
    // After any sendMessage, the regenerate button should be hidden
    lastRemovedWasModelReply = false;
    updateRegenerateButtonVisibility();
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
    if (chatHistory.length === 0 && !systemInstruction) {
        errorMessageDiv.textContent = "No chat history or system instruction to save.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }

    const dataToSave = {
        systemInstruction: systemInstruction,
        chatHistory: chatHistory
    };

    const filename = `gemini_chat_history_${new Date().toISOString().slice(0, 10)}.json`;
    const jsonStr = JSON.stringify(dataToSave, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    errorMessageDiv.textContent = "Chat history and system instruction saved to file.";
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
            const loadedData = JSON.parse(e.target.result);
            // Validate the structure
            if (typeof loadedData !== 'object' || loadedData === null || !Array.isArray(loadedData.chatHistory)) {
                throw new Error("Invalid chat history file format. Expected an object with 'systemInstruction' and an array 'chatHistory'.");
            }

            // Update system instruction
            systemInstruction = loadedData.systemInstruction || '';
            systemInstructionInput.value = systemInstruction;
            setLocalStorageItem('systemInstruction', systemInstruction);

            // Update chat history
            const newChatHistory = loadedData.chatHistory;
            if (!newChatHistory.every(item => item.role && Array.isArray(item.parts))) {
                throw new Error("Invalid chat history entries within the file.");
            }
            chatHistory = newChatHistory;
            
            renderChatHistory(); // This will also call updateRawHistoryInput indirectly
            saveChatHistoryToLocalStorage(); // Save loaded chatHistory to local storage
            errorMessageDiv.textContent = "Chat history and system instruction loaded from file successfully!";
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

// Function to toggle raw chat history visibility
function toggleRawChatHistory() {
    if (rawChatContent) {
        const isHidden = rawChatContent.classList.toggle('hidden');
        setLocalStorageItem('rawChatHistoryHidden', isHidden.toString()); // Save the state
        console.log('Raw chat history visibility saved:', !isHidden);
    }
}

// Function to load raw chat history toggle state from localStorage
function loadRawChatHistoryToggleStateFromLocalStorage() {
    const isHidden = getLocalStorageItem('rawChatHistoryHidden');
    if (isHidden === 'true') {
        rawChatContent.classList.add('hidden');
    } else {
        rawChatContent.classList.remove('hidden'); // Ensure it's shown if 'false' or not set
    }
    console.log('Raw chat history visibility loaded:', isHidden === 'true' ? 'hidden' : 'visible');
}


// Function to update the visibility of the regenerate button
function updateRegenerateButtonVisibility() {
    if (regenerateSystemReplyButton) {
        // Show the button if the last removed entry was a model reply AND there's a user message to regenerate from
        if (lastRemovedWasModelReply && chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            regenerateSystemReplyButton.classList.remove('hidden');
        } else {
            regenerateSystemReplyButton.classList.add('hidden');
        }
    }
}

// Function to remove the last entry from chat history
function removeLastEntry() {
    if (chatHistory.length > 0) {
        const lastEntry = chatHistory.pop(); // Remove the last entry
        lastRemovedWasModelReply = (lastEntry.role === 'model'); // Check if it was a model reply
        renderChatHistory(); // Re-render chat bubbles
        saveChatHistoryToLocalStorage(); // Save updated history
        errorMessageDiv.textContent = `Last entry (${lastEntry.role}) removed.`;
    } else {
        errorMessageDiv.textContent = 'No chat history to remove.';
        lastRemovedWasModelReply = false; // No entry removed, so no model reply removed
    }
    updateRegenerateButtonVisibility(); // Update button visibility after removal
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
}

// Function to regenerate the last system reply
async function regenerateSystemReply() {
    if (!lastRemovedWasModelReply || chatHistory.length === 0 || chatHistory[chatHistory.length - 1].role !== 'user') {
        errorMessageDiv.textContent = 'Cannot regenerate: No previous user message to regenerate from, or last removed was not a model reply.';
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }

    const lastUserMessageText = chatHistory[chatHistory.length - 1].parts[0].text;
    
    // Prepare content for API call, including system instruction
    const conversationContent = [...chatHistory]; // chatHistory already ends with the user message
    if (systemInstruction) {
        conversationContent.unshift({ role: 'user', parts: [{ text: systemInstruction }] });
    }

    await _sendContentToModel(lastUserMessageText, conversationContent);

    lastRemovedWasModelReply = false; // Reset flag after attempting regeneration
    updateRegenerateButtonVisibility(); // Hide button
    adjustTextareaHeight(); // Re-adjust
}


// Function to clear all chat history
function clearAllHistory() {
    if (confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
        chatHistory = []; // Clear the array
        totalInputTokens = 0; // Reset tokens
        totalOutputTokens = 0; // Reset tokens
        currentInputTokens = 0; // Reset current tokens
        currentOutputTokens = 0; // Reset current tokens
        lastRemovedWasModelReply = false; // Reset regeneration state
        renderChatHistory(); // Re-render (will be empty)
        renderTokenStats(); // Update token display
        saveChatHistoryToLocalStorage(); // Save empty history
        saveTokenStatsToLocalStorage(); // Save reset token stats
        errorMessageDiv.textContent = 'All chat history cleared.';
    }
    updateRegenerateButtonVisibility(); // Hide regenerate button
    setTimeout(() => errorMessageDiv.textContent = '', 3000);
}


// Event Listeners
setApiKeyButton.addEventListener('click', setApiKey);
geminiModelSelect.addEventListener('change', updateSelectedModel);
sendMessageButton.addEventListener('click', sendMessage);
applyRawHistoryButton.addEventListener('click', applyRawHistory);
debugButton.addEventListener('click', toggleRawChatHistory); // New: Debug button listener

// System Instruction events
systemInstructionInput.addEventListener('input', saveSystemInstruction);
clearSystemInstructionButton.addEventListener('click', clearSystemInstruction); // New: Clear system instruction button listener

// Chat Save/Load events
saveChatButton.addEventListener('click', downloadChatHistory);
loadChatButton.addEventListener('click', () => loadChatFileInput.click()); // Trigger file input click
loadChatFileInput.addEventListener('change', handleChatFileLoad);

// Chat history action events
removeLastEntryButton.addEventListener('click', removeLastEntry); // New
clearAllHistoryButton.addEventListener('click', clearAllHistory); // New
regenerateSystemReplyButton.addEventListener('click', regenerateSystemReply); // New


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
    loadSelectedModelFromLocalStorage(); // Load selected model
    loadChatHistoryFromLocalStorage(); // Load chat history (or initialize with welcome)
    loadTokenStatsFromLocalStorage(); // Load token stats
    loadRawChatHistoryToggleStateFromLocalStorage(); // Load raw chat toggle state
    
    renderChatHistory(); // Render the initial history (including welcome message) and update raw input
    adjustTextareaHeight(); // Adjust textarea height on page load

    // Set the initial selected model based on dropdown and update global variable
    // geminiModelSelect.value = selectedModel; // This is now handled by loadSelectedModelFromLocalStorage
    updateSelectedModel(); 
    renderTokenStats(); // Render initial token stats
    updateRegenerateButtonVisibility(); // New: Set initial state of regenerate button
});