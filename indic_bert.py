import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ---------------------------------------------------------------------------
#  CONFIG: PASTE YOUR LOCAL FOLDER PATH BELOW
# ---------------------------------------------------------------------------
# Example format: r"C:\Users\Name\Downloads\indic-model" or r"/home/usr/model"
LOCAL_FOLDER_PATH = r"./local_indic_model" 
# ---------------------------------------------------------------------------

# Check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_lang, tgt_lang = "eng_Latn", "hin_Deva"

print(f"Loading model from local folder: {LOCAL_FOLDER_PATH}")

# Load Tokenizer from the local folder
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_FOLDER_PATH, 
    trust_remote_code=True
)

# Load Model from the local folder
model = AutoModelForSeq2SeqLM.from_pretrained(
    LOCAL_FOLDER_PATH, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)

input_sentences = [
    "When I was young, I used to go to the park every day.",
    "We watched a new movie last week, which was very inspiring.",
    "If you had met me at that time, we would have gone out to eat.",
    "My friend has invited me to his birthday party, and I will give him a gift.",
]

batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# Tokenize
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations
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
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# Postprocess
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")