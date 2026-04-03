from transformers import pipeline

# Clinical NER for diseases, procedures, findings
clinical_ner = pipeline(
    "ner",
    model="samrawal/bert-base-uncased_clinical-ner",
    aggregation_strategy="simple"
)

text = '''Patient presents with hypertension and type 2 diabetes. 
        Prescribed metformin 500mg twice daily.'''

entities = clinical_ner(text)
for entity in entities:
    print(entity)