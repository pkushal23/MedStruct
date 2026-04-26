import pandas as pd
import warnings
from tqdm import tqdm

class ClinicalNER:
    def __init__(self, model_name="samrawal/bert-base-uncased_clinical-ner", device=0):
        self.model_name = model_name
        self.device = device
        self.nlp = None

        try:
            from transformers import AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nlp = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=device,
                trust_remote_code=True,
            )
        except Exception as exc:
            self.tokenizer = None
            warnings.warn(
                "ClinicalNER could not be loaded and will be disabled for this run. "
                "Install torch>=2.6 or use a safetensors-backed checkpoint to enable it. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    def extract_entities(self, text):
        if self.nlp is None or not isinstance(text, str) or not text.strip():
            return []
        return self.nlp(text)

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