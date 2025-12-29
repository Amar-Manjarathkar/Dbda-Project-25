import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

# 1. Load the secrets from the .env file
load_dotenv() 

# 2. Get the token variable
hf_token = os.getenv("HF_TOKEN")

# Safety check to ensure the token was found
if hf_token is None:
    raise ValueError("HF_TOKEN not found in .env file. Please create it!")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
save_directory = "./local_indic_model"

print(f"Authenticating with token ending in '...{hf_token[-5:]}'")
print(f"Downloading model to '{save_directory}'...")

try:
    # 3. Pass the token to the tokenizer loader
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=hf_token  # <--- Using the variable from .env
    )
    tokenizer.save_pretrained(save_directory)

    # 4. Pass the token to the model loader
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        token=hf_token  # <--- Using the variable from .env
    )
    model.save_pretrained(save_directory)

    print("Success! Model and tokenizer are saved locally.")

except OSError as e:
    print(f"\nERROR: Access denied. Ensure you accepted the license on HuggingFace.\nError details: {e}")