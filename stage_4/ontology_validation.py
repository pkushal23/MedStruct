import spacy
from scispacy.linking import EntityLinker

class OntologyLinker:
    def __init__(self, model_name="en_core_sci_sm"):
        print(f"Loading SciSpacy model: {model_name} (This may take a minute)...")
        self.nlp = spacy.load(model_name)
        
        print("Adding UMLS Linker to the NLP pipeline...")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def get_umls_concept(self, text):
        # Clean the text to improve matching odds
        text = str(text).lower().strip()
        doc = self.nlp(text)
        
        best_cui = "UNMAPPED"
        best_concept_name = "UNMAPPED"
        highest_score = 0.0

        # Check every entity SciSpacy found in the string
        for ent in doc.ents:
            if ent._.kb_ents:
                # Get the top matching CUI for this specific entity
                cui, score = ent._.kb_ents[0]
                
                # Keep it if it's the highest confidence match in the text
                if score > highest_score:
                    highest_score = score
                    best_cui = cui
                    best_concept_name = self.linker.kb.cui_to_entity[cui].canonical_name
                    
        return best_cui, best_concept_name