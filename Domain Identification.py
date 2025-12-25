import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=os.getenv("HF_TOKEN"))
#### Sentiment Classification Example
text = "This restaurant has amazing food and great service!"

candidate_labels = ["positive", "negative", "neutral"]

result = pipe(text, candidate_labels)
result
#### Topic Classification Example
text = "The latest advancements in AI are transforming healthcare."

candidate_labels = ["technology", "sports", "politics", "entertainment", "finance"]

result = pipe(text, candidate_labels)
result
#### Intent Detection Example
text = "Can you recommend a good book on machine learning?"

candidate_labels = ["book recommendation", "weather inquiry", "restaurant booking", "technical support", "general chat"]

result = pipe(text, candidate_labels)
result
#### Controversial Topic (with multi-label) Example
text = "Climate change is caused by human activities and requires urgent action."

candidate_labels = ["science", "politics", "religion", "sports"]

result = pipe(text, candidate_labels, multi_label=True)  # Allows multiple high scores
result
## Domain Identification (Internal Project)
# from transformers import pipeline

# pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = [
    "Politics",
    "Crime",
    "Military",
    "Terrorism",
    "Radicalisation",
    "Extremism in J&K",
    "Law and Order",
    "Narcotics",
    "Left Wing Extremism",
    "General"
]

text = "Security forces gunned down two Lashkar-e-Taiba militants in an encounter in Pulwama district."

result = pipe(text, candidate_labels)

# Get top 3
top_3_labels = result['labels'][:3]
top_3_scores = result['scores'][:3]

print("Input:", text)
print("Ranked domains:")
for i, (label, score) in enumerate(zip(top_3_labels, top_3_scores), 1):
    print(f"{i}. {label} ({score:.3f})")

# Optional: fallback to "General" if top score is very low
if top_3_scores[0] < 0.30:  # adjust threshold as needed
    print("→ Assigned to: General (low confidence)")
