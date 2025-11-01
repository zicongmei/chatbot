// chatbot.js

let chatHistory = [];
let currentApiKey = '';
let selectedModel = 'gemini-2.5-flash-lite';
let systemInstruction = ''; // New variable for system instruction, reflects current input field content

let totalInputTokens = 0;
let totalOutputTokens = 0;
let currentInputTokens = 0; // New: Tokens for the current request
let currentOutputTokens = 0; // New: Tokens for the current request
let lastRemovedWasModelReply = false; // New: To track if last removed entry was a model reply

// New: Variables to store raw API request/response for debugging
let lastRawRequestBody = null;
let lastRawResponseData = null;


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

// New DOM elements for API Debugging
const showApiDebugButton = document.getElementById('showApiDebugButton');
const apiDebugContent = document.getElementById('apiDebugContent');
const apiRequestBody = document.getElementById('apiRequestBody');
const apiResponseBody = document.getElementById('apiResponseBody');


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

// Function to load system instruction from localStorage (for initial pre-fill)
function loadSystemInstructionFromLocalStorage() {
    const loadedInstruction = getLocalStorageItem('systemInstruction');
    if (loadedInstruction) {
        systemInstruction = loadedInstruction; // Update global variable
        systemInstructionInput.value = loadedInstruction; // Update UI
        console.log('System instruction loaded from local storage.');
    }
}

// Function to clear system instruction from UI, variable, and localStorage
function clearSystemInstruction() {
    if (systemInstructionInput.value.trim() === '') {
        errorMessageDiv.textContent = "Background instruction is already empty.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }
    if (confirm('Are you sure you want to clear the background / system instruction?')) {
        systemInstructionInput.value = '';
        systemInstruction = ''; // Ensure global variable is also cleared
        setLocalStorageItem('systemInstruction', ''); // Directly clear from local storage
        updateRawHistoryInput(); // Update raw history display
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
    // and there's no system instruction already providing context, add the instruction message.
    if (chatHistory.length === 0 && !systemInstruction) {
        const initialInstruction = `Your task is to write the messages in this chat/roleplay.
  Use *asterisks* for actions, and (parentheses) for the internal thought processes of a character.
  NEVER try to "wrap up" the roleplay. This is a never-ending roleplay.
  Multi-line messages are not allowed - each individual message must be a single paragraph.
  Avoid unnecessary and unoriginal repetition of previous messages.
  Write the next 10 messages - remember to make them interesting, authentic, descriptive, natural, engaging, and creative.
  Use the same language as input or previous diaglog`;
        chatHistory.push({ role: 'model', parts: [{ text: initialInstruction }] });
        console.log('Initialized chat history with the roleplay instruction.');
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
                systemInstruction: systemInstruction, // Use the global variable which reflects the input
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
        systemInstruction = newSystemInstruction; // Update global variable
        systemInstructionInput.value = systemInstruction; // Update UI
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
        
        // Handle multiple parts within a message, joining them
        const textContent = msg.parts.map(part => part.text).join('\n'); // Join parts with newline
        const paragraph = document.createElement('p');
        paragraph.textContent = textContent;
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

    // Clear previous raw API debug data before a new request
    lastRawRequestBody = null;
    lastRawResponseData = null;
    apiRequestBody.textContent = 'No API request made yet.';
    apiResponseBody.textContent = 'No API response received yet.';


    try {
        const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`;

        const requestBody = {
            contents: contentToSendForAPI, // This will be the actual history for the API call, potentially with appended system instruction part
            generationConfig: {
                maxOutputTokens: 5000,
            },
        };

        // Store the raw request body before sending
        lastRawRequestBody = JSON.stringify(requestBody, null, 2);


        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': currentApiKey,
            },
            body: lastRawRequestBody, // Use the stored stringified body
        });

        if (!response.ok) {
            const errorData = await response.json();
            // Store the raw error response
            lastRawResponseData = JSON.stringify(errorData, null, 2);
            const errorMessage = errorData.error ? errorData.error.message : response.statusText;
            throw new Error(`API Error: ${errorMessage} (Status: ${response.status})`);
        }

        const data = await response.json();
        // Store the raw successful response
        lastRawResponseData = JSON.stringify(data, null, 2);
        
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

    // Get current system instruction from the input field (one-shot for this request)
    const currentSystemInstruction = systemInstructionInput.value.trim();

    // Add user message to history
    chatHistory.push({ role: 'user', parts: [{ text: userMessageText }] });
    renderChatHistory(); // Render the new user message and update raw history input
    saveChatHistoryToLocalStorage(); // Save updated history
    messageInput.value = ''; // Clear input
    adjustTextareaHeight(); // Reset textarea height

    // Prepare content for API call: Start with a copy of the chat history
    const conversationContent = [...chatHistory];

    // Append system instruction to the parts of the *last user message* in the conversation content for API
    if (currentSystemInstruction !== '') {
        const lastMessageIndex = conversationContent.length - 1;
        // Ensure the last message is a user message before appending
        if (lastMessageIndex >= 0 && conversationContent[lastMessageIndex].role === 'user') {
            conversationContent[lastMessageIndex].parts.push({ text: `\nSYSTEM INSTRUCTION: ${currentSystemInstruction}` });
        }
    }

    const success = await _sendContentToModel(userMessageText, conversationContent);

    if (!success) {
        // If API call failed, remove the last user message from history
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
            // Note: If system instruction was appended, it was to a copy, not the actual chatHistory entry.
            // So simply popping removes the user message correctly.
            chatHistory.pop();
            renderChatHistory(); // Re-render to reflect removal and update raw history input
            saveChatHistoryToLocalStorage(); // Save updated history
        }
    }

    // Clean up system instruction: Clear the input field and associated variables/storage
    // This makes the system instruction a "one-shot" instruction per request.
    if (currentSystemInstruction !== '') { // Only clear if there was content
        systemInstructionInput.value = ''; // Clear UI
        systemInstruction = ''; // Clear global variable
        setLocalStorageItem('systemInstruction', ''); // Clear from local storage
        updateRawHistoryInput(); // Update raw history display
        errorMessageDiv.textContent = "Background instruction applied and cleared.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
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
    if (chatHistory.length === 0 && systemInstruction.trim() === '') {
        errorMessageDiv.textContent = "No chat history or system instruction to save.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
        return;
    }

    const dataToSave = {
        systemInstruction: systemInstruction, // Use the current value of the global variable
        chatHistory: chatHistory
    };

    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    const timestamp = `${year}-${month}-${day}_${hours}-${minutes}-${seconds}`;
    const filename = `gemini_chat_history_${timestamp}.json`;
    
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
            systemInstruction = loadedData.systemInstruction || ''; // Update global variable
            systemInstructionInput.value = systemInstruction; // Update UI
            setLocalStorageItem('systemInstruction', systemInstruction); // Save to local storage

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

// Function to toggle raw API request/response visibility
function toggleApiDebugDisplay() {
    if (apiDebugContent.classList.contains('hidden')) { // If it's about to be shown
        apiRequestBody.textContent = lastRawRequestBody || 'No API request made yet. Send a message to see the request.';
        apiResponseBody.textContent = lastRawResponseData || 'No API response received yet. Send a message to see the response.';
    }
    apiDebugContent.classList.toggle('hidden');
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
    
    // Get current system instruction from the input field (one-shot for this request)
    const currentSystemInstruction = systemInstructionInput.value.trim();

    // Prepare content for API call, including system instruction
    const conversationContent = [...chatHistory]; // chatHistory already ends with the user message

    // Append system instruction to the parts of the *last user message* in the conversation content for API
    if (currentSystemInstruction !== '') {
        const lastMessageIndex = conversationContent.length - 1;
        // Ensure the last message is a user message before appending
        if (lastMessageIndex >= 0 && conversationContent[lastMessageIndex].role === 'user') {
            conversationContent[lastMessageIndex].parts.push({ text: `system content: ${currentSystemInstruction}` });
        }
    }

    await _sendContentToModel(lastUserMessageText, conversationContent);

    // Clean up system instruction if it was used
    // This makes the system instruction a "one-shot" instruction per request.
    if (currentSystemInstruction !== '') {
        systemInstructionInput.value = ''; // Clear UI
        systemInstruction = ''; // Clear global variable
        setLocalStorageItem('systemInstruction', ''); // Clear from local storage
        updateRawHistoryInput(); // Update raw history display
        errorMessageDiv.textContent = "Background instruction applied and cleared.";
        setTimeout(() => errorMessageDiv.textContent = '', 3000);
    }

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
        lastRawRequestBody = null; // Clear raw API debug data
        lastRawResponseData = null; // Clear raw API debug data
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

// New: API Debug button listener
showApiDebugButton.addEventListener('click', toggleApiDebugDisplay);


// System Instruction events
// Keep the global 'systemInstruction' variable in sync with the input field for UI display (e.g., raw history)
systemInstructionInput.addEventListener('input', () => {
    systemInstruction = systemInstructionInput.value.trim();
    updateRawHistoryInput();
});
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
    loadSystemInstructionFromLocalStorage(); // Load system instruction (pre-fills UI and global variable)
    loadSelectedModelFromLocalStorage(); // Load selected model
    loadChatHistoryFromLocalStorage(); // Load chat history (or initialize with welcome)
    loadTokenStatsFromLocalStorage(); // Load token stats
    loadRawChatHistoryToggleStateFromLocalStorage(); // Load raw chat toggle state
    
    renderChatHistory(); // Render the initial history (including welcome message) and update raw input
    adjustTextareaHeight(); // Adjust textarea height on page load

    // Set the initial selected model based on dropdown and update global variable
    updateSelectedModel(); 
    renderTokenStats(); // Render initial token stats
    updateRegenerateButtonVisibility(); // New: Set initial state of regenerate button
});