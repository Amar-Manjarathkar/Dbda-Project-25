# Use a pipeline as a high-level helper
from transformers import pipeline
from dotenv import load_dotenv
import os
load_dotenv()
recognize = pipeline("token-classification", model="dslim/bert-base-NER", token=os.getenv("HF_TOKEN"))
from transformers import pipeline
import os

# Your token is loaded from environment variable
# recognize = pipeline("token-classification", model="dslim/bert-base-NER", token=os.getenv("HF_TOKEN"))

# Optional: group entities (recommended for cleaner output)
recognize = pipeline("token-classification", model="dslim/bert-base-NER", 
                     token=os.getenv("HF_TOKEN"), aggregation_strategy="simple")
### Internal Project
##### Example 1
text = "Prime Minister Narendra Modi visited Washington D.C. to meet President Joe Biden and discuss trade with Apple and Google executives."

results = recognize(text)

for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")
##### Example 2

text = "The United Nations headquarters in New York hosted a summit attended by Elon Musk and Satya Nadella from Microsoft."

results = recognize(text)
for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")
##### Example 3
text = "Omar Abdullah addressed a rally in Srinagar, criticizing the BJP government over Article 370 revocation."

results = recognize(text)
for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")
##### Example 4
text = "Security forces arrested Hizbul Mujahideen commander Ryaz Naikoo in Pulwama district of Jammu and Kashmir."

results = recognize(text)
for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")
##### Example 5
text = "Tesla CEO Elon Musk announced a new factory in Shanghai, China, in partnership with Tata Motors from India."

results = recognize(text)
for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")
