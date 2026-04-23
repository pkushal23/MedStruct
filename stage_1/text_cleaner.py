import re
import pandas as pd

# Expanded abbreviation dictionary for common MIMIC-IV shorthand
LRABR_DICT = {
    r'\bHTN\b': 'hypertension',
    r'\bDM2\b': 'type 2 diabetes mellitus',
    r'\bCHF\b': 'congestive heart failure',
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

def clean_clinical_text(text: str) -> str:
    """Applies the 5-step cleaning sequence defined in MedStruct 1.4."""
    if not isinstance(text, str):
        return ""
    
    # 1. Scrub blanks/de-id marks completely so they don't leak into NER
    text = re.sub(r'_{3,}', ' ', text)
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    
    # 2. Expand abbreviations via LRABR dictionary
    for abbrev, expansion in LRABR_DICT.items():
        text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
    # 3. Normalize lab value formats (e.g., "Sodium: 140 mEq/L" -> "LABTEST: 140 mEq/L")
    text = re.sub(r'([A-Za-z]+)\:\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)', r'LABTEST \1: \2 \3', text)
    
    # 4. Final Scrub: Remove any residual explicit [DEID] strings from legacy formatting
    text = re.sub(r'\[DEID\]', ' ', text)
    
    # 5. Clean up extra whitespace left behind by the scrubbing
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    return text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning to the entire dataframe."""
    df_clean = df.copy()
    if 'text' in df_clean.columns:
        df_clean['text'] = df_clean['text'].apply(clean_clinical_text)
    return df_clean