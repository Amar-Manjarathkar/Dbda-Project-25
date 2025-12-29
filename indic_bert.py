# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit.processor import IndicProcessor

# # ---------------------------------------------------------------------------
# #  CONFIG: PASTE YOUR LOCAL FOLDER PATH BELOW
# # ---------------------------------------------------------------------------
# # Example format: r"C:\Users\Name\Downloads\indic-model" or r"/home/usr/model"
# LOCAL_FOLDER_PATH = r"Code\local_indic_model" 
# # ---------------------------------------------------------------------------

# # Check device
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# src_lang, tgt_lang = "eng_Latn", "hin_Deva"

# print(f"Loading model from local folder: {LOCAL_FOLDER_PATH}")

# # Load Tokenizer from the local folder
# tokenizer = AutoTokenizer.from_pretrained(
#     LOCAL_FOLDER_PATH, 
#     trust_remote_code=True
# )

# # Load Model from the local folder
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     LOCAL_FOLDER_PATH, 
#     use_cache=False,
#     trust_remote_code=True, 
#     torch_dtype=torch.float16, 
#     attn_implementation="flash_attention_2"
# )

# ip = IndicProcessor(inference=True)

# input_sentences = [
#     "When I was young, I used to go to the park every day.",
#     "We watched a new movie last week, which was very inspiring.",
#     "If you had met me at that time, we would have gone out to eat.",
#     "My friend has invited me to his birthday party, and I will give him a gift.",
# ]

# batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# # Tokenize
# # inputs = tokenizer(
# #     batch,
# #     truncation=True,
# #     padding="longest",
# #     return_tensors="pt",
# #     return_attention_mask=True,
# # )
# inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

# # Generate translations
# with torch.no_grad():
#     generated_tokens = model.generate(
#         **inputs,
#         use_cache=True,
#         min_length=0,
#         max_length=256,
#         num_beams=5,
#         num_return_sequences=1,
#     )

# # Decode
# generated_tokens = tokenizer.batch_decode(
#     generated_tokens,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )

# # Postprocess
# translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

# for input_sentence, translation in zip(input_sentences, translations):
#     print(f"{src_lang}: {input_sentence}")
#     print(f"{tgt_lang}: {translation}")

# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit.processor import IndicProcessor

# # ---------------------------------------------------------------------------
# #  CONFIG
# # ---------------------------------------------------------------------------
# LOCAL_FOLDER_PATH = r"Code\local_indic_model" 
# # ---------------------------------------------------------------------------

# # Check device
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running on device: {DEVICE}")

# src_lang, tgt_lang = "eng_Latn", "hin_Deva"

# print(f"Loading model from local folder: {LOCAL_FOLDER_PATH}")

# # 1. Load Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     LOCAL_FOLDER_PATH, 
#     trust_remote_code=True
# )

# # 2. Load Model (SAFE MODE)
# # CRITICAL: Do NOT use device_map, low_cpu_mem_usage, or torch_dtype here.
# # Those arguments create "Meta" (ghost) tensors which crash custom models.
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     LOCAL_FOLDER_PATH, 
#     trust_remote_code=True,
#     use_cache=False,  # Keep this False to avoid shape errors
# )

# # 3. Manually move to GPU
# print("Moving model to GPU...")
# # model.to(DEVICE)

# # 4. Optional: Convert to Half Precision (Speed Boost)
# # We do this AFTER loading to avoid the "Meta Tensor" error
# if DEVICE == "cuda":
#     print("Converting to float16...")
#     model.half()

# model.eval() 

# ip = IndicProcessor(inference=True)

# input_sentences = [
#     "When I was young, I used to go to the park every day.",
#     "We watched a new movie last week, which was very inspiring.",
#     "If you had met me at that time, we would have gone out to eat.",
#     "My friend has invited me to his birthday party, and I will give him a gift.",
# ]

# print("Preprocessing inputs...")
# batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# # Tokenize
# inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

# # 5. Move inputs to the same device as the model
# inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# print("Generating translations...")

# # Generate translations
# with torch.no_grad():
#     generated_tokens = model.generate(
#         **inputs,
#         use_cache=True,
#         min_length=0,
#         max_length=256,
#         num_beams=5,
#         num_return_sequences=1,
#     )

# # Decode
# # Move tokens back to CPU for decoding
# generated_tokens = tokenizer.batch_decode(
#     generated_tokens.detach().cpu().tolist(), 
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )

# # Postprocess
# translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

# print("\n" + "="*50)
# for input_sentence, translation in zip(input_sentences, translations):
#     print(f"Input ({src_lang}): {input_sentence}")
#     print(f"Output ({tgt_lang}): {translation}")
#     print("-" * 50)
# print("="*50)

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
LOCAL_FOLDER_PATH = r"Code\local_indic_model" 
# ---------------------------------------------------------------------------

# 1. FORCE CPU MODE (To bypass the GPU/Meta bugs)
# Since your model refused to move to CUDA in the logs, we will stay on CPU.
DEVICE = "cpu"
print(f"--> Running on device: {DEVICE}")

src_lang, tgt_lang = "eng_Latn", "hin_Deva"

print(f"--> Loading model from: {LOCAL_FOLDER_PATH}")

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_FOLDER_PATH, 
    trust_remote_code=True
)

# 3. Load Model
# We disable 'low_cpu_mem_usage' to ensure real weights are loaded, not "Meta" ghosts.
model = AutoModelForSeq2SeqLM.from_pretrained(
    LOCAL_FOLDER_PATH, 
    trust_remote_code=True,
    use_cache=False, 
    low_cpu_mem_usage=False, 
    device_map=None
)

# 4. Ensure Model is on CPU
model.to(DEVICE)
model.eval()

# NOTE: We REMOVED 'model.half()'. 
# Float16 is not supported for LayerNorm on CPU and causes crashes.

ip = IndicProcessor(inference=True)

input_sentences = [
    "When I was young, I used to go to the park every day.",
    "We watched a new movie last week, which was very inspiring.",
    "If you had met me at that time, we would have gone out to eat.",
    "My friend has invited me to his birthday party, and I will give him a gift.",
]

print("--> Preprocessing inputs...")
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# Tokenize
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

# 5. Align Inputs to CPU
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

print("--> Generating translations (this might take a few seconds)...")

# Generate
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode
generated_tokens = tokenizer.batch_decode(
    generated_tokens.detach().cpu().tolist(), 
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# Postprocess
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

print("\n" + "="*50)
for input_sentence, translation in zip(input_sentences, translations):
    print(f"Input: {input_sentence}")
    print(f"Output: {translation}")
    print("-" * 50)
print("="*50)