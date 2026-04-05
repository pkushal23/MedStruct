import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

class ClinicalNER:
    def __init__(self, model_name="samrawal/bert-base-uncased_clinical-ner", device=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)

    def extract_entities(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        tokens = self.tokenizer.encode(text, truncation=True, max_length=510, add_special_tokens=False)
        safe_text = self.tokenizer.decode(tokens)
        return self.nlp(safe_text)

    def process_dataframe(self, df, text_col='sentence_text'):
        results = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing ClinicalBERT"):
            entities = self.extract_entities(row[text_col])
            for ent in entities:
                results.append({
                    'note_id': row['note_id'],
                    'hadm_id': row['hadm_id'],
                    'sentence_index': row['sentence_index'],
                    'entity_group': ent.get('entity_group', ent.get('entity')),
                    'word': ent['word'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'score': float(ent['score']),
                    'source_model': 'ClinicalBERT'
                })
        return pd.DataFrame(results)