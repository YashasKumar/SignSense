from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA 2 model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Change this to another size if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Define words to be rearranged
words = ["I", "head", "pain"]

# Create a prompt for sentence formation
prompt = f"Form a grammatically correct sentence using these words: {', '.join(words)}"

# Tokenize and generate output
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=50)

# Decode and print the result
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
