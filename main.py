from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# model_name = "01-ai/Yi-Coder-9B-Chat"
# model_name = "openbmb/MiniCPM3-4B"
model_name ="mattshumer/Reflection-Llama-3.1-70B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

print("loaded model.")

conversation_history = []

while True:
    # Get the input data from the user
    input_text = input("> ")

    conversation_history.append({"role": "user", "content": input_text})
    # messages = [
    #     {"role": "user", "content": "推荐5个北京的景点。"},
    # ]
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
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    output_token_ids = [
        outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]

    responses = tokenizer.batch_decode(
        output_token_ids, skip_special_tokens=True)[0]
    print("== ", responses)
    # lines = responses.split("\n")
    # print("-- response has {} lines. ".format(len(lines)), lines)

    # Add interaction to conversation history
    conversation_history.append({"role": "bot", "content": responses})

    # print("== ", lines[-1])

