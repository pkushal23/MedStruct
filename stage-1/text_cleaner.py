import re
import pandas as pd

    
LRABR_DICT = {
    r'\bHTN\b': 'hypertension',
    r'\bDM2\b': 'type 2 diabetes mellitus',
    r'\bCHF\b': 'congestive heart failure',
    r'\bpt\b': 'patient',
    r'\bdx\b': 'diagnosis',
    r'\bhx\b': 'history'
}

def clean_clinical_text(text: str) -> str:
    """Applies the 5-step cleaning sequence defined in MedStruct 1.4."""
    if not isinstance(text, str):
        return ""
    
    # 1. Replace blanks/de-id marks with [DEID]
    text = re.sub(r'_{3,}', '[DEID]', text)
    text = re.sub(r'\[\*\*.*?\*\*\]', '[DEID]', text)
    
    # 2. Expand abbreviations via LRABR dictionary
    for abbrev, expansion in LRABR_DICT.items():
        text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
    # 3. Normalize lab value formats (e.g., "Sodium: 140 mEq/L" -> "LABTEST: 140 mEq/L")
    # This is a basic regex; you may need to tune it for specific MIMIC lab formats
    text = re.sub(r'([A-Za-z]+)\:\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)', r'LABTEST \1: \2 \3', text)
    
    # 4. Normalize Whitespace (reduce multi-newlines and multi-spaces)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def clean_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Wrapper to clean a whole dataframe."""
    df_clean = df.copy()
    df_clean['cleaned_text'] = df_clean[text_col].apply(clean_clinical_text)
    return df_clean