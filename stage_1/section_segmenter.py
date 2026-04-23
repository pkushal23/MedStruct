import re
import pandas as pd

# Canonical mappings defined in MedStruct - UPDATED FOR MIMIC-IV FLEXIBILITY
DS_SECTIONS = {
    "Chief Complaint": [r"(?i)chief complaint\s*:?", r"(?i)c/c\s*:?"],
    "HPI": [r"(?i)history of present illness\s*:?", r"(?i)hpi\s*:?"],
    "PMH": [r"(?i)past medical history\s*:?", r"(?i)pmh\s*:?"],
    "Medications": [r"(?i)discharge medications\s*:?", r"(?i)medications on admission\s*:?"],
    "Assessment/Plan": [r"(?i)assessment and plan\s*:?", r"(?i)a/p\s*:?"],
    "Discharge Dx": [r"(?i)discharge diagnos(is|es)\s*:?"],
    "Hospital Course": [r"(?i)brief hospital course\s*:?", r"(?i)hospital course\s*:?"]
}

RR_SECTIONS = {
    "Indication": [r"(?i)indication\s*:?", r"(?i)reason for exam\s*:?"],
    "Findings": [r"(?i)findings\s*:?"],
    "Impression": [r"(?i)impression\s*:?", r"(?i)conclusions\s*:?"]
}

def extract_sections(text: str, note_type: str) -> dict:
    """
    Extracts canonical sections based on note type (discharge vs radiology).
    Returns a dictionary of {canonical_name: section_text}.
    """
    if not isinstance(text, str):
        return {"full_text": ""}

    normalized_note_type = str(note_type).strip().lower()
    
    if normalized_note_type == 'discharge':
        target_sections = DS_SECTIONS
    elif normalized_note_type == 'radiology':
        target_sections = RR_SECTIONS
    else:
        return {"full_text": text}

    print(f"[DEBUG] extract_sections: note_type='{note_type}' -> normalized='{normalized_note_type}'")
    print(f"[DEBUG] Using sections: {list(target_sections.keys())}")
    
    extracted = {}
    current_section = "full_text"
    current_text = []

    # Split text by lines to parse headers
    lines = text.split('\n')
    print(f"[DEBUG] Text has {len(lines)} lines")
    
    for line_idx, line in enumerate(lines):
        matched_section = None
        # Check if the line matches any known section header
        for sec_name, patterns in target_sections.items():
            for pattern in patterns:
                # Use re.match to ensure the header is at the start of the line
                if re.match(pattern, line.strip()):
                    matched_section = sec_name
                    print(f"[DEBUG] Line {line_idx}: '{line[:50]}...' matched section '{sec_name}'")
                    break
            if matched_section:
                break
        
        if matched_section:
            # Save the previous section
            if current_text:
                extracted[current_section] = "\n".join(current_text).strip()
            # Start tracking the new section
            current_section = matched_section
            current_text = []
        else:
            current_text.append(line)

    # Catch the final section
    if current_text:
        extracted[current_section] = "\n".join(current_text).strip()

    print(f"[DEBUG] Final extracted sections: {[(k, len(v)) for k, v in extracted.items()]}")
    return extracted

def segment_dataframe(df: pd.DataFrame, note_type: str) -> pd.DataFrame:
    """Applies section extraction to the entire dataframe."""
    records = []
    for _, row in df.iterrows():
        sections_dict = extract_sections(row.get('text', ''), note_type)
        print(f"[DEBUG] segment_dataframe: Extracted sections: {list(sections_dict.keys())}")
        for sec_name, sec_text in sections_dict.items():
            if sec_text and len(sec_text) > 0:
                print(f"[DEBUG]   Section '{sec_name}': {len(sec_text)} chars")
            # Only keep sections that actually have text
            if sec_text.strip():
                record = row.to_dict()
                record['section_name'] = sec_name
                record['section_text'] = sec_text
                records.append(record)
                
    return pd.DataFrame(records)