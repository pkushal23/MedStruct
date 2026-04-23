import pandas as pd
from tqdm import tqdm
import warnings
import re
from dotenv import load_dotenv

try:
    # Works when imported as part of the stage_3 package (e.g., uvicorn loading test.py)
    from stage_3.proximity_rules import get_cooccurring_entities
except ModuleNotFoundError:
    # Fallback for direct script execution from inside stage_3
    from proximity_rules import get_cooccurring_entities

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class RelationExtractor:
    def __init__(self, drug_labels, disease_labels, model_name="facebook/bart-large-mnli", device=0):
        self.drug_labels = drug_labels
        self.disease_labels = disease_labels
        self.classifier = None

        try:
            from transformers import pipeline

            print(f"Loading Zero-Shot Classifier: {model_name} on GPU (device {device})...")
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        except Exception as exc:
            warnings.warn(
                "Zero-shot relation classifier could not be loaded. "
                "Falling back to keyword-based relation extraction. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    def _keyword_fallback_relation(self, context):
        text = (context or "").lower()

        treats_patterns = [
            r"\btreated with\b",
            r"\bstarted on\b",
            r"\bgiven\b",
            r"\bprescribed\b",
            r"\bfor\b",
            r"\bimproved on\b",
            r"\bresponded to\b",
        ]
        causes_patterns = [
            r"\bcaused\b",
            r"\bdue to\b",
            r"\bsecondary to\b",
            r"\badverse\b",
            r"\bside effect\b",
            r"\binduced\b",
            r"\bcomplication\b",
        ]

        if any(re.search(pattern, text) for pattern in causes_patterns):
            return "CAUSES", 0.68
        if any(re.search(pattern, text) for pattern in treats_patterns):
            return "TREATS", 0.66
        return "UNRELATED", 0.0

    def extract_drug_disease(self, df_entities, df_sentences, window, valid_sections=None, threshold=0.85):
        all_types = self.drug_labels + self.disease_labels
        co_occurring = get_cooccurring_entities(
            df_entities=df_entities,
            df_sentences=df_sentences,
            target_types=all_types,
            max_sentence_window=window,
            valid_sections=valid_sections,
        )

        is_drug_1 = co_occurring['entity_group_1'].isin(self.drug_labels)
        is_disease_2 = co_occurring['entity_group_2'].isin(self.disease_labels)
        is_disease_1 = co_occurring['entity_group_1'].isin(self.disease_labels)
        is_drug_2 = co_occurring['entity_group_2'].isin(self.drug_labels)

        candidates = co_occurring[(is_drug_1 & is_disease_2) | (is_disease_1 & is_drug_2)].copy()

        if candidates.empty:
            print("No valid candidates found within the specified window and sections.")
            return pd.DataFrame()

        sentence_map = df_sentences.set_index(['note_id', 'sentence_index'])['sentence_text'].to_dict()

        contexts = []
        valid_indices = []
        
        for idx, row in candidates.iterrows():
            text_context = sentence_map.get((row['note_id'], row['sentence_index_1']), "")
            if row['sentence_index_1'] != row['sentence_index_2']:
                text_context += " " + sentence_map.get((row['note_id'], row['sentence_index_2']), "")

            drug  = row['word_1'] if row['entity_group_1'] in self.drug_labels else row['word_2']
            disease = row['word_2'] if row['entity_group_1'] in self.drug_labels else row['word_1']
            
            contexts.append(text_context)
            valid_indices.append(idx)

        verified_results = []

        if self.classifier is not None:
            # --- CHUNKED BATCHING WITH PROGRESS BAR ---
            results = []
            batch_size = 16

            print(f"Running inference on {len(contexts)} candidates in batches of {batch_size}...")

            for i in tqdm(range(0, len(contexts), batch_size), desc="Classifying Relations"):
                batch_contexts = contexts[i:i + batch_size]

                batch_results = self.classifier(
                    batch_contexts,
                    # Speak like a doctor to catch ADEs
                    candidate_labels=["is used to treat", "resulted in an adverse side effect or complication called", "is completely unrelated to"],
                    # The zero-shot pipeline expects exactly one positional placeholder: {}
                    hypothesis_template="Based on the text, {}.",
                    batch_size=batch_size,
                )

                # The pipeline returns a single dict if the batch size happens to be exactly 1 at the end
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]

                results.extend(batch_results)

            # --- MAP RESULTS BACK TO DATAFRAME ---
            for i, res in enumerate(results):
                top_label = res['labels'][0]
                top_score = res['scores'][0]

                if top_label == "is used to treat":
                    mapped_label = "TREATS"
                elif top_label == "resulted in an adverse side effect or complication called":
                    mapped_label = "CAUSES"
                else:
                    mapped_label = "UNRELATED"

                if mapped_label in ["TREATS", "CAUSES"] and top_score >= threshold:
                    original_row = candidates.loc[valid_indices[i]].copy()
                    original_row['relation_type'] = mapped_label
                    original_row['model_confidence'] = top_score
                    verified_results.append(original_row)
        else:
            print("Running keyword fallback relation extraction...")
            for i, context in enumerate(contexts):
                mapped_label, score = self._keyword_fallback_relation(context)
                if mapped_label in ["TREATS", "CAUSES"] and score >= threshold:
                    original_row = candidates.loc[valid_indices[i]].copy()
                    original_row['relation_type'] = mapped_label
                    original_row['model_confidence'] = score
                    verified_results.append(original_row)

        return pd.DataFrame(verified_results)