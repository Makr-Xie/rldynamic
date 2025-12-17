import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-8B"
local_model_dir = "/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/models/Qwen3-8B"

os.makedirs(local_model_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_model_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_model_dir)
