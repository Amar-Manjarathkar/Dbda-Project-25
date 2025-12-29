import os
from transformers import pipeline

# ----------------------------------------------------------------------
# IMPORTANT: DEFINE YOUR LOCAL PATH HERE
# ----------------------------------------------------------------------
LOCAL_MODEL_PATH = r"C:\Users\amarn\OneDrive\Desktop\CDAC Project\Code\models"


# --- List of Strings to Test (Multiple Languages) ---
STRING_LIST_TO_TEST = [
    "मुझे हिंदी सीखना है और यह मॉडल बहुत अच्छा है।",   # Hindi
    "La plume de ma tante est sur la table.",          # French
    "De trein is al vertrokken van het station.",      # Dutch
    "Where is the nearest coffee shop?",               # English
    "இந்த மாதிரி மொழி அடையாளம் காண முடியுமா?",            # Tamil
    "Tengo que go to the store to buy some bread."    # Code-switching (Spanish/English)
]


def run_batch_language_identification(local_path: str, text_list: list):
    """
    Loads the model and runs language identification on an entire list of strings
    in a single, optimized batch operation.
    """
    
    if not os.path.isdir(local_path):
        print(f"🚨 ERROR: Local model directory not found at: {local_path}")
        return

    # 1. Load the Pipeline (Same as before)
    print(f"Loading model from local directory: {local_path}...")
    try:
        language_detector = pipeline(
            "text-classification",  
            model=local_path,       
            tokenizer=local_path    
        )
        print("✅ Model loaded successfully for batch prediction.")
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL. Error Details: {e}")
        return

    # 2. Run the Prediction on the entire list
    print("\n--- Running Batch Language Identification ---")
    
    # PASS THE ENTIRE LIST TO THE PIPELINE
    results = language_detector(text_list)
    
    # 3. Print the Output
    print("\n--- Batch Prediction Results ---")
    
    # Iterate through both the original input and the prediction results
    for i, (input_text, result) in enumerate(zip(text_list, results)):
        label = result.get('label', 'UNKNOWN')
        score = result.get('score', 0.0) * 100
        
        print(f"--- Input #{i+1} ---")
        print(f"  Input:    \"{input_text}\"")
        print(f"  Language: **{label.upper()}**")
        print(f"  Confidence: {score:.2f}%")
        print("-" * 30)


# --- Execute the function ---
run_batch_language_identification(LOCAL_MODEL_PATH, STRING_LIST_TO_TEST)