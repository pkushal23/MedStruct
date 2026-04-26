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
    def __init__(self, drug_labels, disease_labels, threshold=0.75, model_name="facebook/bart-large-mnli", device=0):
        self.drug_labels = drug_labels
        self.disease_labels = disease_labels
        self.threshold = threshold
        self.classifier = None
        self.device = -1
        self.relation_hypotheses = {
            "TREATS": "is a medication used specifically to treat",
            "CAUSES": "is a direct side effect or complication caused by",
            "UNRELATED": "is completely unrelated and mentioned in a different context",
        }
        self.evidence_hypotheses = {
            "exact synonym": "The sentence expresses an exact synonym relationship between the clinical terms.",
            "clinical abbreviation": "The sentence links a clinical abbreviation to its expanded meaning.",
            "direct implication": "The sentence directly implies this clinical relationship.",
            "hyponym/subtype": "The sentence indicates one term is a subtype or specific kind of the other.",
        }

        try:
            import torch
            from transformers import pipeline

            if device >= 0 and torch.cuda.is_available() and torch.cuda.device_count() > device:
                self.device = device
                device_desc = f"GPU (device {self.device}: {torch.cuda.get_device_name(self.device)})"
            else:
                self.device = -1
                device_desc = "CPU"
                if device >= 0:
                    warnings.warn(
                        "CUDA device requested for relation extraction but not available in the active torch build. "
                        "Falling back to CPU. Install a CUDA-enabled torch wheel to run on GPU.",
                        RuntimeWarning,
                    )

            print(f"Loading Zero-Shot Classifier: {model_name} on {device_desc}...")
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)
        except Exception as exc:
            warnings.warn(
                "Zero-shot relation classifier could not be loaded. "
                "Falling back to keyword-based relation extraction. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    def _classify_relation(self, context, drug, disease):
        if self.classifier is None:
            return "UNRELATED", 0.0

        candidate_hypotheses = {
            label: f"The medication {drug} {template} the condition {disease}."
            for label, template in self.relation_hypotheses.items()
        }

        result = self.classifier(
            context,
            candidate_labels=list(candidate_hypotheses.values()),
            hypothesis_template="{}",
        )

        top_hypothesis = result['labels'][0]
        top_score = float(result['scores'][0])
        mapped_label = next(
            (label for label, hypothesis in candidate_hypotheses.items() if hypothesis == top_hypothesis),
            "UNRELATED",
        )
        return mapped_label, top_score

    def _classify_evidence_type(self, context):
        if self.classifier is None:
            return "direct implication", 0.55

        result = self.classifier(
            context,
            candidate_labels=list(self.evidence_hypotheses.values()),
            hypothesis_template="{}",
        )
        top_hypothesis = result['labels'][0]
        top_score = float(result['scores'][0])
        evidence_type = next(
            (label for label, hypothesis in self.evidence_hypotheses.items() if hypothesis == top_hypothesis),
            "direct implication",
        )
        return evidence_type, top_score

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

    def extract_drug_disease(self, df_entities, df_sentences, window, valid_sections=None, threshold=None, threshold_by_section=None):
        default_threshold = self.threshold if threshold is None else threshold

        all_types = self.drug_labels + self.disease_labels
        candidates = get_cooccurring_entities(
            df_entities=df_entities,
            df_sentences=df_sentences,
            target_types=all_types,
            max_sentence_window=window,
            valid_sections=valid_sections,
        )

        if candidates.empty:
            print("No valid candidates found within the specified window and sections.")
            return pd.DataFrame()

        sentence_map = df_sentences.set_index(['note_id', 'sentence_index'])['sentence_text'].to_dict()

        verified_results = []

        if self.classifier is not None:
            print(f"Running directional inference on {len(candidates)} candidates...")

            for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Classifying Relations"):
                type_1 = row['entity_group_1']
                type_2 = row['entity_group_2']

                is_valid_pair = (
                    (type_1 in self.drug_labels and type_2 in self.disease_labels) or
                    (type_1 in self.disease_labels and type_2 in self.drug_labels)
                )

                if not is_valid_pair:
                    continue

                drug = row['word_1'] if type_1 in self.drug_labels else row['word_2']
                disease = row['word_2'] if type_1 in self.drug_labels else row['word_1']

                text_context = sentence_map.get((row['note_id'], row['sentence_index_1']), "")
                if row['sentence_index_1'] != row['sentence_index_2']:
                    text_context += " " + sentence_map.get((row['note_id'], row['sentence_index_2']), "")

                section_name = row.get('section_name', 'unknown')
                effective_threshold = (
                    float(threshold_by_section(section_name))
                    if callable(threshold_by_section)
                    else default_threshold
                )

                mapped_label, top_score = self._classify_relation(text_context, drug, disease)

                if mapped_label in ["TREATS", "CAUSES"] and top_score >= effective_threshold:
                    evidence_type, evidence_confidence = self._classify_evidence_type(text_context)
                    original_row = row.copy()
                    original_row['relation_type'] = mapped_label
                    original_row['model_confidence'] = top_score
                    original_row['evidence_type'] = evidence_type
                    original_row['evidence_confidence'] = evidence_confidence
                    verified_results.append(original_row)
        else:
            print("Running keyword fallback relation extraction...")
            for _, row in candidates.iterrows():
                type_1 = row['entity_group_1']
                type_2 = row['entity_group_2']

                is_valid_pair = (
                    (type_1 in self.drug_labels and type_2 in self.disease_labels) or
                    (type_1 in self.disease_labels and type_2 in self.drug_labels)
                )

                if not is_valid_pair:
                    continue

                text_context = sentence_map.get((row['note_id'], row['sentence_index_1']), "")
                if row['sentence_index_1'] != row['sentence_index_2']:
                    text_context += " " + sentence_map.get((row['note_id'], row['sentence_index_2']), "")

                section_name = row.get('section_name', 'unknown')
                effective_threshold = (
                    float(threshold_by_section(section_name))
                    if callable(threshold_by_section)
                    else default_threshold
                )

                mapped_label, score = self._keyword_fallback_relation(text_context)
                if mapped_label in ["TREATS", "CAUSES"] and score >= effective_threshold:
                    evidence_type = "direct implication"
                    evidence_confidence = 0.55
                    original_row = row.copy()
                    original_row['relation_type'] = mapped_label
                    original_row['model_confidence'] = score
                    original_row['evidence_type'] = evidence_type
                    original_row['evidence_confidence'] = evidence_confidence
                    verified_results.append(original_row)

        return pd.DataFrame(verified_results)