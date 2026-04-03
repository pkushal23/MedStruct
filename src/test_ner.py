from src.fetch_notes import fetch_one_note
from src.preprocess import clean_text
from src.ner_module import extract_medical_entities

df = fetch_one_note()

text = clean_text(df.iloc[0]["text"])
entities = extract_medical_entities(text)

print("=== EXTRACTED MEDICAL ENTITIES ===\n")

for e in entities[:50]:
    print(e)