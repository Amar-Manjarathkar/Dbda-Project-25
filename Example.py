# git clone https://huggingface.co/ai4bharat ai4bharat/indic-bert
# cd ai4bharat/indic-bert

# pip3 install transformers
# pip3 install sentencepiece

from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
# model = AutoModel.from_pretrained('ai4bharat/indic-bert')

# text = "भारत एक महान देश है।"
# inputs = tokenizer(text, return_tensors='pt')

from transformers import pipeline
translator = pipeline("translation", model="ai4bharat/indictrans2-indic-en-distilbert", device=0)  # GPU=0 or -1=CPU
result = translator("नमस्ते, आप कैसे हैं?")  # Output: [{'translation_text': 'Hello, how are you?'}]
summarizer = pipeline("summarization", model="ai4bharat/indicbart-xlsum")
summary = summarizer("लंबा हिंदी लेख यहाँ...")  # Indic summary
print(result)
# outputs = model(**inputs)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
print(transliterate("नमस्ते", sanscript.DEVANAGARI, sanscript.IAST))  # namaste
# print(outputs)
# # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))