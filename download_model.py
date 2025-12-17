from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Replace with the specific model you are using
model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
local_dir = "./indictrans_model"

# Download and save
print("Downloading and saving model locally...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print(f"Model saved to {local_dir}")