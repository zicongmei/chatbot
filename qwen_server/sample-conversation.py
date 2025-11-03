import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# You may need to install the dependencies: pip install torch transformers accelerate
model_name = "Qwen/Qwen3-8B" 

print(f"Loading model: {model_name}...")

# load the tokenizer and the model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using 'cuda' if available, otherwise 'cpu'. 
    # 'device_map="auto"' is generally preferred for large models.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" 
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please ensure you have required libraries (torch, transformers, accelerate) and enough GPU/CPU resources.")
    exit()

# Initialize the conversation history
# The model will use this list to understand the context of the conversation.
messages = []

print("\n--- Qwen CLI Chat Started ---")
print("Type 'exit' or 'quit' to end the conversation.")
print("-" * 35)

# Infinite loop for continuous conversation
while True:
    # 1. Get user input
    user_input = input("You: ")
    
    # Check for exit command
    if user_input.lower() in ['exit', 'quit']:
        print("--- Chat ended. Goodbye! ---")
        break
    
    # 2. Append the new user message to the history
    messages.append({"role": "user", "content": user_input})
    
    # 3. Apply the chat template to the full history
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    
    # 4. Prepare model input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 5. Conduct text completion
    print("AI: (Thinking...) ", end="", flush=True) # Give feedback while generating
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048, # Limiting token generation for speed
        do_sample=True, # Often better for chat
        temperature=0.7 # Common setting for balanced creativity
    )
    
    # 6. Extract the generated text (only the new part)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 7. Decode and separate thinking content (Qwen-specific parsing)
    thinking_content = ""
    content = ""
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        # If </think> is not found, treat everything as content
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
    # 8. Print the model's response and thinking (optional)
    if thinking_content:
        print(f"\nThinking: {thinking_content}")
        print("-" * 35) # Separator for clarity
        print(f"AI: {content}")
    else:
        print(f"\rAI: {content}") # Use \r to overwrite the (Thinking...) message
    
    # 9. Append the model's response to the history for context in the next turn
    # Only append the actual content, not the thinking part, to the history.
    messages.append({"role": "assistant", "content": content})

    print("-" * 35)