# git clone https://huggingface.co/ai4bharat ai4bharat/indic-bert
# cd ai4bharat/indic-bert

# pip3 install transformers
# pip3 install sentencepiece

from transformers import AutoModel, AutoTokenizer

# --- Start of Corrected/Uncommented Section for Indic-BERT ---
# Load the model and tokenizer for Indic-BERT
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
# model = AutoModel.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert', 
                                  use_safetensors=True, 
                                  trust_remote_code=True)

text = "भारत एक महान देश है।"
# Tokenize the input text, generating token IDs
inputs = tokenizer(text, return_tensors='pt')

# Print the model outputs (if desired)
outputs = model(**inputs)
# Print the token IDs converted back to tokens (This is what you asked about)
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])) 
# --- End of Corrected/Uncommented Section ---


# Keep the other parts for Translation, Summarization, and Transliteration
from transformers import pipeline
# Using 'device=0' for GPU or '-1' for CPU ensures local execution
# translator = pipeline("translation", model="ai4bharat/indictrans2-indic-en-distilbert", device=-1)
# result = translator("नमस्ते, आप कैसे हैं?")
# summarizer = pipeline("summarization", model="ai4bharat/indicbart-xlsum", device=-1)
# summary = summarizer("लंबा हिंदी लेख यहाँ...")
# print(result)

translator = pipeline("translation", 
                      model="ai4bharat/indictrans2-all-distilbert", 
                      device=-1, 
                      trust_remote_code=True,
                      batch_size=1) # Ensure single-item processing for low-latency
result = translator("नमस्ते, आप कैसे हैं?")  # Output: [{'translation_text': 'Hello, how are you?'}]
print(result)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
print(transliterate("नमस्ते", sanscript.DEVANAGARI, sanscript.IAST))
print(outputs) # Outputs tensor is typically not printed unless needed for debugging