import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# 1. Point to your local folder path
model_path = r"C:\Users\amarn\OneDrive\Desktop\CDAC Project\indictrans_model"

# 2. Load from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
ip = IndicProcessor(inference=True)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Now use it as usual
input_sentences = ["India is a great country.", "I love programming in Python."]
src_lang, tgt_lang = "eng_Latn", "sat_Olck"

batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, num_beams=5, max_length=256, use_cache=False)

translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
final_translations = ip.postprocess_batch(translations, lang=tgt_lang)

print(final_translations)

