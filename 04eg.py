from transformers import pipeline

# 1. Sentiment Analysis
# Loads a default sentiment analysis model (usually distilbert-base-uncased-finetuned-sst-2-english)
classifier = pipeline("sentiment-analysis")
results = classifier("I love using Hugging Face models!")
print(results)
# Output: [{'label': 'POSITIVE', 'score': 0.9998...}]


# 2. Named Entity Recognition (NER)
# Automatically identifies people, organizations, and locations
ner_pipe = pipeline("ner", grouped_entities=True) # Note: 'grouped_entities' might prompt a deprecation warning
text = "Hugging Face Inc. is based in New York City."
entities = ner_pipe(text)
print(entities)
# Output might look like: [{'entity_group': 'ORG', 'word': 'Hugging Face Inc.', ...}, {'entity_group': 'LOC', 'word': 'New York City', ...}]


# 3. Text Generation
# Loads a default text generation model (usually gpt2)
generator = pipeline("text-generation", model="distilgpt2")
prompt = "In this course, we will teach you how to"
generated_text = generator(prompt, max_length=50, num_return_sequences=1)
print(generated_text[0]['generated_text'])
