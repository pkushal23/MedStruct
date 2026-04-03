import spacy
import re

# Load biomedical NER model
nlp = spacy.load("en_ner_bc5cdr_md")

# Clinical abbreviation expansion
ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "id-t2dm": "insulin-dependent type 2 diabetes mellitus",
    "bph": "benign prostatic hyperplasia",
    "lfts": "liver function tests",
    "rcc": "renal cell carcinoma",
    "n/v": "nausea vomiting",
    "abd": "abdominal"
}

def expand_abbreviations(text: str) -> str:
    for short, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text, flags=re.IGNORECASE)
    return text

def clean_entity(entity_text: str) -> str:
    # Remove dosage info like "50 mg", "2 ml"
    entity_text = re.sub(
        r"\b\d+(\.\d+)?\s?(mg|ml|mcg|g|units?)\b",
        "",
        entity_text,
        flags=re.IGNORECASE
    )

    entity_text = re.sub(r"\s+", " ", entity_text).strip()
    return entity_text

def extract_medical_entities(text: str):
    text = expand_abbreviations(text)
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        if ent.label_ in ["CHEMICAL", "DISEASE"]:
            cleaned = clean_entity(ent.text)

            if cleaned:
                entities.append({
                    "entity": cleaned,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

    # Deduplicate
    seen = set()
    deduped = []
    for e in entities:
        key = (e["entity"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped