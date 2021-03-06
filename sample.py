import time
import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch
import transformers

#Will need at least 13-14GB of Vram for CUDA
if torch.cuda.is_available():
    print("CUDA")
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
else:
    print("NOT CUDA")
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model.eval()

input_text = "Hello my name is Blake and"
input_ids = tokenizer.encode(str(input_text), return_tensors='pt')

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=20,
    top_p=0.7,
    top_k=0,
    temperature=1.0,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
