from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
# ⚠️ Replace 'YOUR_ACCESS_TOKEN_HERE' with the token you copied in Step 1
HF_TOKEN = "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" 

# Example Model: Indic-BERT (Requires token as it is gated)
MODEL_ID = "ai4bharat/indic-bert"

## 1. Using AutoTokenizer and AutoModel
try:
    # Pass the token directly to the from_pretrained method
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModel.from_pretrained(MODEL_ID,use_safetensors=True, token=HF_TOKEN)
    
    print(f"✅ Successfully loaded {MODEL_ID} using the token.")
    
    # Example usage:
    text = "नमः शिवाय"
    inputs = tokenizer(text, return_tensors='pt')
    # outputs = model(**inputs) 
    print(f"Tokenized input IDs: {inputs['input_ids'][0]}")

except Exception as e:
    print(f"❌ An error occurred: {e}")



## 2. Using the pipeline function (e.g., for Translation)

# # Example: IndicTrans2 Translation Model
# TRANSLATION_MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"

# # try:
#     # The pipeline function also accepts the token argument
#     # translator = pipeline("translation", 
#     #                       model=TRANSLATION_MODEL_ID, 
#     #                       token=HF_TOKEN,
#     #                       trust_remote_code=True)
    
# #     result = translator("कल मैं दिल्ली जा रहा हूँ।")
# #     print(f"\n✅ Translation Result: {result}")
    
# # except Exception as e:
# #     print(f"\n❌ An error occurred during translation: {e}")
# try:
#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     translator = pipeline("translation", 
#                           model=TRANSLATION_MODEL_ID, 
#                           token=HF_TOKEN,
#                           trust_remote_code=True,
#                           max_length=400,
#                           model_kwargs={"torch_dtype": torch.float32})
#     # 1. Hindi (hin_Deva) to English (eng_Latn)
#     source_text = "नमस्ते, आप कैसे हैं?"  
    
#     # 2. Prepend the language tags to the text:
#     tagged_input = f"hin_Deva eng_Latn {source_text}"
    
#     # The pipeline will now receive the correctly tagged input:
#     result = translator(tagged_input)  
#     print("✅ Translation Result:", result)
    
# except Exception as e:
#     print(f"❌ An error occurred during translation: {e}")

# TRANSLATION_MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M" # Use the 200M parameter model (~914 MB)
# try:
#     # Set the device and ensure trust_remote_code/token are passed
#     translator = pipeline("translation", 
#                           model=TRANSLATION_MODEL_ID, 
#                           device=-1, # Switch to CPU temporarily to avoid GPU memory issues with the 4.46 GB model
#                           trust_remote_code=True,
#                           token=HF_TOKEN,
#                           model_kwargs={"torch_dtype": torch.float32}) 
    
#     # Use the correct language tags
#     source_text = "नमस्ते, आप कैसे हैं?"  
#     tagged_input = f"hin_Deva eng_Latn {source_text}"
    
#     result = translator(tagged_input)  
#     print("✅ Translation Result (Fast Model):", result)
    
# except Exception as e:
#     print(f"❌ An error occurred during optimized translation: {e}")

# --- Translation (Manual Loading for Stability) ---
TRANSLATION_MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
try:
    # 1. Setup Device and Data Type
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For stability, we'll try FP32 (default) on CPU and FP16 on GPU (if available)
    DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

    # 2. Load Model and Tokenizer Manually
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL_ID, 
        trust_remote_code=True, 
        token=HF_TOKEN,
        # Load model with correct precision and move to device
        torch_dtype=DTYPE, 
        use_safetensors=True
    ).to(DEVICE)
    
    # 3. Prepare Input with Language Tags (as a single string)
    source_text = "नमस्ते, आप कैसे हैं?"  
    tagged_input = f"hin_Deva eng_Latn {source_text}"

    # 4. Tokenize and Generate (Manual Steps)
    inputs = tokenizer(tagged_input, return_tensors="pt").to(DEVICE)

    # Note: IndicTrans2 models handle language IDs internally via the input string
    generated_tokens = model.generate(
        **inputs,
        num_return_sequences=1,
        num_beams=5,
        max_length=256
    )

    # 5. Decode the Result
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print("✅ Translation Result (Stable Method):", result)
    
except Exception as e:
    print(f"❌ An error occurred during stable translation: {e}")