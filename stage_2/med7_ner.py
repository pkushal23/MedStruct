import pandas as pd
import spacy

class Med7NER:
    def __init__(self, model_name="en_core_med7_lg"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        doc = self.nlp(text)
        return doc.ents

    def process_dataframe(self, df, text_col='sentence_text'):
        results = []
        for _, row in df.iterrows():
            entities = self.extract_entities(row[text_col])
            for ent in entities:
                results.append({
                    'note_id': row['note_id'],
                    'hadm_id': row['hadm_id'],
                    'sentence_index': row['sentence_index'],
                    'entity_group': ent.label_,
                    'word': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'score': 1.0,
                    'source_model': 'Med7'
                })
        return pd.DataFrame(results)