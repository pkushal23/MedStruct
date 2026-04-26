import pandas as pd

try:
    import spacy
except Exception:  # pragma: no cover - spaCy is optional at runtime
    spacy = None


NOISE_SINGLE_TOKENS = {'and', 'by', 'repair'}
NOISE_POS_TAGS = {'CCONJ', 'ADP', 'VERB', 'AUX'}
LOW_INFO_STOPWORDS = {
    'a', 'an', 'the', 'all', 'this', 'that', 'these', 'those', 'other',
    'some', 'any', 'each', 'every', 'either', 'neither',
}
TARGET_LABELS_FOR_LOW_INFO_FILTER = {'treatment', 'problem'}
UNITS_REQUIRING_DRUG = {'strength', 'dosage'}
DRUG_LABEL = 'drug'

DISEASE_MARKERS = {
    'itis', 'osis', 'emia', 'infection', 'sepsis', 'pneumonia',
    'cancer', 'failure', 'syndrome', 'fracture', 'stroke',
    'hemorrhage', 'tumor', 'disease',
}
TREATMENT_CUES = {
    'therapy', 'treatment', 'management', 'surgery', 'procedure',
    'infusion', 'medication', 'dose',
}


def _load_pos_tagger():
    if spacy is None:
        return None

    for model_name in ('en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'):
        try:
            return spacy.load(model_name, disable=['ner', 'parser', 'lemmatizer'])
        except Exception:
            continue

    return None


def _is_noise_entity(word, nlp):
    normalized = str(word).strip().lower()
    if not normalized:
        return True

    if ' ' in normalized:
        return False

    if normalized in NOISE_SINGLE_TOKENS:
        return True

    if nlp is None:
        return False

    doc = nlp(normalized)
    if not doc:
        return False

    token = doc[0]
    return token.pos_ in NOISE_POS_TAGS


def _is_low_information_label_term(word, label, nlp):
    if label not in TARGET_LABELS_FOR_LOW_INFO_FILTER:
        return False

    normalized = str(word).strip().lower()
    if not normalized or ' ' in normalized:
        return False

    if normalized in LOW_INFO_STOPWORDS:
        return True

    if nlp is None:
        return False

    doc = nlp(normalized)
    if not doc:
        return False
    return bool(doc[0].is_stop)


def _looks_like_disease_term(word):
    normalized = str(word).strip().lower()
    if not normalized:
        return False

    has_disease_marker = any(marker in normalized for marker in DISEASE_MARKERS)
    has_treatment_cue = any(cue in normalized for cue in TREATMENT_CUES)
    return has_disease_marker and not has_treatment_cue


def _normalize_treatment_problem_labels(df):
    labels = df['entity_group_norm']
    treatment_mask = labels.eq('treatment')
    disease_like_mask = df['word'].apply(_looks_like_disease_term)
    should_relabel = treatment_mask & disease_like_mask
    if should_relabel.any():
        df.loc[should_relabel, 'entity_group'] = 'problem'
        df.loc[should_relabel, 'entity_group_norm'] = 'problem'
    return df


def _drop_unlinked_unit_entities(df):
    drug_keys = set(
        df.loc[df['entity_group_norm'].eq(DRUG_LABEL), ['note_id', 'sentence_index']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if not drug_keys:
        return df[~df['entity_group_norm'].isin(UNITS_REQUIRING_DRUG)]

    is_unit = df['entity_group_norm'].isin(UNITS_REQUIRING_DRUG)
    has_drug_same_sentence = df[['note_id', 'sentence_index']].apply(
        lambda row: (row['note_id'], row['sentence_index']) in drug_keys,
        axis=1,
    )
    return df[~is_unit | has_drug_same_sentence]

def final_polish(df):
    """
    Final data scientist's cleanup:
    1. Removes persistent subword artifacts.
    2. Drops single-token POS noise like conjunctions, prepositions, and verbs.
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    pos_tagger = _load_pos_tagger()
    df_clean['entity_group_norm'] = df_clean['entity_group'].astype(str).str.strip().str.lower()
    df_clean = df_clean[~df_clean['word'].astype(str).str.contains('##', na=False)]
    df_clean = df_clean[df_clean['word'].astype(str).str.len() > 2]
    df_clean = df_clean[~df_clean['entity_group_norm'].str.contains('label', na=False)]
    df_clean = df_clean[~df_clean['word'].apply(lambda word: _is_noise_entity(word, pos_tagger))]
    df_clean = df_clean[
        ~df_clean.apply(
            lambda row: _is_low_information_label_term(row['word'], row['entity_group_norm'], pos_tagger),
            axis=1,
        )
    ]

    df_clean = _normalize_treatment_problem_labels(df_clean)
    df_clean = _drop_unlinked_unit_entities(df_clean)

    df_clean['word'] = df_clean['word'].astype(str).str.lower().str.strip()
    df_clean = df_clean.sort_values(['note_id', 'sentence_index', 'word'])
    df_clean = df_clean.drop(columns=['entity_group_norm'])
    
    return df_clean


if __name__ == "__main__":
    input_path = '../data/processed/entities.csv'
    output_path = '../data/processed/entities_refined.csv'
    
    print(f"Loading raw entities from {input_path}...")
    try:
        df_entities = pd.read_csv(input_path)
        print(f"Original entities count: {len(df_entities)}")
        
        clean_entities = final_polish(df_entities)
        
        clean_entities.to_csv(output_path, index=False)
        print(f"Polished down to {len(clean_entities)} high-quality entities.")
        print(f"Saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Run stage 2 first.")