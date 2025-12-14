# C:\Users\amarn\OneDrive\Desktop\CDAC Project\Code\models

from transformers import pipeline

# Specify the custom directory for this one download
custom_path = "C:\\Users\\amarn\\OneDrive\\Desktop\\CDAC Project\\Code\\models" 

language_classifier = pipeline(
    "text-classification", 
    model="papluca/xlm-roberta-base-language-detection",
    # Pass the custom path here
    cache_dir=custom_path
)

print(f"Model files are saved to: {custom_path}")