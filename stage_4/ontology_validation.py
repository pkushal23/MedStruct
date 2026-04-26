import spacy
import re
from scispacy.linking import EntityLinker

# Expanded LRABR_DICT to include common MIMIC-IV abbreviations
LRABR_DICT = {
    r'\bHTN\b': 'hypertension',
    r'\bDM2\b': 'type 2 diabetes mellitus',
    r'\bCHF\b': 'congestive heart failure',
    r'\bivf\b': 'intravenous fluids',
    r'\bct\b': 'computed tomography',
    r'\bpt\b': 'patient',
    r'\bdx\b': 'diagnosis',
    r'\bhx\b': 'history',
    r'\bs/p\b': 'status post',
    r'\bc/o\b': 'complains of',
    r'\bprn\b': 'as needed',
    r'\bBID\b': 'twice a day',
    r'\bTID\b': 'three times a day',
    r'\bQID\b': 'four times a day',
    r'\bsob\b': 'shortness of breath'
}

class OntologyLinker:
    def __init__(self, model_name="en_core_sci_lg"):
        """
        Initialize SciSpacy with a fallback mechanism for environment compatibility.
        """
        self.model_name = model_name
        model_candidates = [model_name]
        if model_name != "en_core_sci_sm":
            model_candidates.append("en_core_sci_sm")

        last_error = None
        for candidate in model_candidates:
            try:
                print(f"Loading SciSpacy model: {candidate} (This may take a minute)...")
                self.nlp = spacy.load(candidate)

                print("Adding UMLS Linker to the NLP pipeline...")
                self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                self.linker = self.nlp.get_pipe("scispacy_linker")
                self.model_name = candidate

                if candidate != model_name:
                    print("WARNING: Falling back to 'en_core_sci_sm' because the requested model could not initialize.")
                return
            except Exception as exc:
                last_error = exc
                print(f"Failed to initialize '{candidate}': {exc}")

        raise last_error

    def _expand_abbreviations(self, text):
        """Pre-expand abbreviations using LRABR_DICT before UMLS linking."""
        expanded_text = text
        for abbrev, expansion in LRABR_DICT.items():
            expanded_text = re.sub(abbrev, expansion, expanded_text, flags=re.IGNORECASE)
        return expanded_text

    def _semantic_vector_similarity(self, text_a, text_b):
        """Compute cosine similarity using SciSpacy token vectors."""
        doc_a = self.nlp.make_doc(str(text_a).lower().strip())
        doc_b = self.nlp.make_doc(str(text_b).lower().strip())

        if not doc_a.vector_norm or not doc_b.vector_norm:
            return 0.0

        try:
            return float(doc_a.similarity(doc_b))
        except Exception:
            return 0.0

    def get_umls_concept(self, text):
        """
        Resolves text to CUI with a confidence threshold. 
        Low-confidence results return 'UNMAPPED' to trigger higher-tier LLM rescue.
        """
        text_str = str(text).lower().strip()
        
        # Step 1: Expand abbreviations
        expanded_text = self._expand_abbreviations(text_str)
        
        # Step 2: Try direct linking with expanded text
        doc = self.nlp(expanded_text)
        
        if not doc.ents or not doc.ents[0]._.kb_ents:
            return "UNMAPPED", "UNMAPPED"
            
        # Step 3: Get the top candidate and apply a precision threshold
        best_cui, score = doc.ents[0]._.kb_ents[0]
        
        # A threshold of 0.65 ensures high-precision mappings. 
        # Anything lower is better handled by the LLM Fallback Agent.
        if score < 0.65:
            return "UNMAPPED", "UNMAPPED"
            
        canonical_name = self.linker.kb.cui_to_entity[best_cui].canonical_name
        return best_cui, canonical_name