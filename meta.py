# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path


model_name = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda"

access_token = Path("token").read_text().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    token=access_token)

print("loaded model.")

conversation_history = []

while True:
    # Get the input data from the user
    input_text = input("> ")

    conversation_history.append({"role": "user", "content": input_text})

    # Tokenize the input text and history
    model_inputs = tokenizer.apply_chat_template(conversation_history,
                                                 return_tensors="pt",
                                                 add_generation_prompt=False).to(device)

    # Generate the response from the model
    outputs = model.generate(model_inputs,
                             max_new_tokens=1024,
                             top_p=0.7,
                             temperature=0.7,
                             pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    output_token_ids = [
        outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]

    responses = tokenizer.batch_decode(
        output_token_ids, skip_special_tokens=True)[0]
    print("== ", responses)
    conversation_history.append({"role": "bot", "content": responses})

