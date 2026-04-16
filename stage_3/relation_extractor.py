import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from .proximity_rules import get_cooccurring_entities
from dotenv import load_dotenv

load_dotenv()

class RelationExtractor:
    def __init__(self, drug_labels, disease_labels, model_name="valhalla/distilbart-mnli-12-3", device=-1):
        self.drug_labels = drug_labels
        self.disease_labels = disease_labels
        print(f"Loading Zero-Shot Classifier: {model_name}...")
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    def extract_drug_disease(self, df_entities, df_sentences, window, threshold=0.6):
        all_types = self.drug_labels + self.disease_labels
        co_occurring = get_cooccurring_entities(df_entities, all_types, window)

        is_drug_1 = co_occurring['entity_group_1'].isin(self.drug_labels)
        is_disease_2 = co_occurring['entity_group_2'].isin(self.disease_labels)
        is_disease_1 = co_occurring['entity_group_1'].isin(self.disease_labels)
        is_drug_2 = co_occurring['entity_group_2'].isin(self.drug_labels)

        candidates = co_occurring[(is_drug_1 & is_disease_2) | (is_disease_1 & is_drug_2)].copy()

        if candidates.empty:
            return pd.DataFrame()

        print(f"Verifying {len(candidates)} candidate relationships...")
        verified_results = []

        sentence_map = df_sentences.set_index(['note_id', 'sentence_index'])['sentence_text'].to_dict()

        for _, row in tqdm(candidates.iterrows(), total=len(candidates)):
            text_context = sentence_map.get((row['note_id'], row['sentence_index_1']), "")
            if row['sentence_index_1'] != row['sentence_index_2']:
                text_context += " " + sentence_map.get((row['note_id'], row['sentence_index_2']), "")

            if row['entity_group_1'] == 'DRUG':
                drug, disease = row['word_1'], row['word_2']
            else:
                drug, disease = row['word_2'], row['word_1']

            templ = f"This text shows that {drug} treating {disease} is {{}}."

            res = self.classifier(
                text_context, 
                candidate_labels=["true", "false"],
                hypothesis_template=templ
            )

            confidence = res['scores'][res['labels'].index("true")]

            if confidence >= threshold:
                row['relation_type'] = 'TREATS_OR_CAUSES'
                row['model_confidence'] = confidence
                verified_results.append(row)

        return pd.DataFrame(verified_results)