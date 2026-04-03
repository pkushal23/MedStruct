from src.fetch_notes import fetch_one_note
from src.preprocess import clean_text
from src.ner_module import extract_medical_entities
from src.verify_module import get_prescriptions, verify_drugs

df = fetch_one_note()

row = df.iloc[0]
hadm_id = row["hadm_id"]

text = clean_text(row["text"])
entities = extract_medical_entities(text)

prescriptions = get_prescriptions(hadm_id)
verified = verify_drugs(entities, prescriptions)

print("=== PRESCRIPTIONS FROM STRUCTURED TABLE ===\n")
for p in prescriptions[:30]:
    print(p)

print("\n=== VERIFIED DRUGS ===\n")
for v in verified:
    print(v)