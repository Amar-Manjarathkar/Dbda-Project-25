from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# --- Configuration ---
MODEL_ID = "papluca/xlm-roberta-base-language-detection" 
# Replace with the model you need (e.g., "bert-base-uncased")
LOCAL_SAVE_PATH = r"C:\Users\amarn\OneDrive\Desktop\CDAC Project\Code\models"

# ---------------------

def download_and_save_model(model_id: str, save_path: str):
    """Downloads a model and tokenizer from the Hugging Face Hub and saves them locally."""
    print(f"Starting download for: {model_id}")
    
    # 1. Create the local directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    try:
        # 2. Load the model and tokenizer (This triggers the download)
        # Note: We use AutoModelFor... because the language ID model is a classification model.
        # Use AutoModel for base models, AutoModelForCausalLM for generation, etc.
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Model and Tokenizer loaded successfully into cache.")

        # 3. Save the downloaded files from cache to the specified local path
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print("\n✅ Download and Save Complete!")
        print(f"All files are now available in: {save_path}")
        print("You can now use this path to load the model offline.")

    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Ensure the MODEL_ID is correct and you have an active internet connection.")

# --- Execute the download ---
download_and_save_model(MODEL_ID, LOCAL_SAVE_PATH)