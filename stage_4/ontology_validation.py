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
        
        # If no entities found
        if not doc.ents or not doc.ents[0]._.kb_ents:
            return "UNMAPPED", "UNMAPPED"
        
        # Extract the highest scoring Concept Unique Identifier (CUI)
        cui, score = doc.ents[0]._.kb_ents[0]
        concept = self.linker.kb.cui_to_entity[cui]
        
        return cui, concept.canonical_name