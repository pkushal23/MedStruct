import re
import pandas as pd

# Canonical mappings defined in MedStruct
DS_SECTIONS = {
    "Chief Complaint": [r"CHIEF COMPLAINT\s*:", r"C/C\s*:"],
    "HPI": [r"HISTORY OF PRESENT ILLNESS\s*:", r"HPI\s*:"],
    "PMH": [r"PAST MEDICAL HISTORY\s*:", r"PMH\s*:"],
    "Medications": [r"DISCHARGE MEDICATIONS\s*:", r"MEDICATIONS ON ADMISSION\s*:"],
    "Assessment/Plan": [r"ASSESSMENT AND PLAN\s*:", r"A/P\s*:"],
    "Discharge Dx": [r"DISCHARGE DIAGNOSIS\s*:", r"DISCHARGE DIAGNOSES\s*:"]
}

RR_SECTIONS = {
    "Indication": [r"INDICATION\s*:", r"REASON FOR EXAM\s*:"],
    "Findings": [r"FINDINGS\s*:"],
    "Impression": [r"IMPRESSION\s*:", r"CONCLUSIONS\s*:"]
}

def extract_sections(text: str, note_type: str) -> dict:
    """
    Extracts canonical sections based on note type (discharge vs radiology).
    Returns a dictionary of {canonical_name: section_text}.
    """
    if not isinstance(text, str):
        return {"full_text": ""}
        
    section_map = DS_SECTIONS if note_type == 'DS' else RR_SECTIONS
    extracted = {}
    
    for canonical_name, patterns in section_map.items():
        for pattern in patterns:
            # Look for the header, grab everything until the next All-Caps header or end of string
            regex = rf"{pattern}\s*(.*?)(?=\n[A-Z\s/]+:|$)"
            match = re.search(regex, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                extracted[canonical_name] = match.group(1).strip()
                break # Found the section, move to the next canonical name
                
    # Fallback to full_text if no sections matched (Step 1.3 spec)
    if not extracted:
        extracted["full_text"] = text.strip()
        
    return extracted

def segment_dataframe(df: pd.DataFrame, note_type: str) -> pd.DataFrame:
    """Applies section segmentation and flattens the dictionary into columns."""
    df_segmented = df.copy()
    
    # Apply extraction
    section_dicts = df_segmented['cleaned_text'].apply(lambda x: extract_sections(x, note_type))
    
    # Expand dictionary keys into new dataframe columns
    sections_df = pd.DataFrame(section_dicts.tolist(), index=df_segmented.index)
    
    # Merge back
    return pd.concat([df_segmented, sections_df], axis=1).drop(columns=['text', 'cleaned_text'])