import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import numpy as np
import scipy.io as sio

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    dtype=torch.float16, 
    device_map="auto"
)

# Enable hidden states output
model.generation_config.output_hidden_states = True
model.generation_config.return_dict_in_generate = True

msgs = [{"role": "user", "content": "Say hello in five words."}]

# 1. Get the formatted string first
text = tokenizer.apply_chat_template(
    msgs, 
    tokenize=False, 
    add_generation_prompt=True
)

# 2. Tokenize into a proper dictionary of tensors
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 3. Unpack the dictionary (**inputs) into generate
out = model.generate(
    **inputs, 
    max_new_tokens=20, 
)



# Print results
generated_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]

pprint(out.hidden_states)

print("Generated Text:", tokenizer.decode(generated_tokens))
print("Hidden states extracted:", len(out.hidden_states), len(out.hidden_states[0]), out.hidden_states[0][0].shape)

n_layers = model.config.num_hidden_layers
LAYERS = [round(n_layers * f) for f in (0.25, 0.5, 0.75, 0.95)]

# Extract activations only for the newly generated tokens
acts = np.stack([
    np.stack([step[l][0, -1, :].float().cpu().numpy() for l in LAYERS])
    for step in out.hidden_states
]) # shape: [n_tokens, n_layers, hidden_dim]

# Save for MATLAB bake-off (USING DUMMY VALUES FOR SMOKE TEST)
sio.savemat("run_eh.mat", {
    "acts": acts.astype(np.float16),
    "violated": 0,           # Dummy value
    "onset": -1,             # Dummy value
    "difficulty": "solvable",# Dummy value
    "condition": "control"   # Dummy value
})