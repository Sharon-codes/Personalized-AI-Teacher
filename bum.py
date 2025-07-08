import kagglehub

# Download latest version
path = kagglehub.model_download("qwen-lm/qwen-3/transformers/1.7b")

# Import transformers for model loading and inference
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer from the downloaded path
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

def ask_ai(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the question from the answer if echoed
    if answer.startswith(question):
        answer = answer[len(question):].strip()
    return answer.strip()

print("Ask your question (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = ask_ai(user_input)
    print("AI:", response)