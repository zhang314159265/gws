from transformers import GPT2Tokenizer, GPT2Model
import torch

torch.set_default_device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
text = "Give me an brief introduction to gpt2."
encoded_input = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = torch.compile(model)(**encoded_input)

# print(output)
