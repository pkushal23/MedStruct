import pandas as pd


INTERVENTION_LABELS = {'drug', 'treatment'}
OBSERVATION_LABELS = {'problem', 'test'}


def _is_plausible_pair(type_1, type_2):
    t1 = str(type_1).strip().lower()
    t2 = str(type_2).strip().lower()
    return (
        (t1 in INTERVENTION_LABELS and t2 in OBSERVATION_LABELS)
        or (t1 in OBSERVATION_LABELS and t2 in INTERVENTION_LABELS)
    )


def get_cooccurring_entities(df_entities, df_sentences, target_types, max_sentence_window, valid_sections=None):
    if not isinstance(target_types, (list, tuple, set, pd.Series)):
        raise TypeError(f"target_types must be list-like, got {type(target_types).__name__}")

    label_col = 'entity_group' if 'entity_group' in df_entities.columns else 'entity_type'

    # Filter only for the entity types we care about
    df_filtered = df_entities[df_entities[label_col].isin(target_types)].copy()
    
    # Merge the dataframe with itself to create pairs of entities
    merged = pd.merge(
        df_filtered, 
        df_filtered, 
        on=['note_id', 'hadm_id'], 
        suffixes=('_1', '_2')
    )

    # Clinical gate: only intervention<->observation pairs are meaningful for relation extraction.
    merged = merged[
        merged.apply(
            lambda row: _is_plausible_pair(row[label_col + '_1'], row[label_col + '_2']),
            axis=1,
        )
    ]
    
    # THE FIX: Enforce a strict ordering so A->B is kept, but B->A is dropped
    merged = merged[merged['start_1'] < merged['start_2']]
    
    # Calculate how many sentences apart they are
    merged['sentence_diff'] = (merged['sentence_index_1'] - merged['sentence_index_2']).abs()
    
    # Keep only the pairs that fall within our window
    valid_relations = merged[merged['sentence_diff'] <= max_sentence_window].copy()
    
    # Always attach section_name so downstream stages can group and score by section,
    # even when valid_sections is disabled for relaxed fallback extraction.
    if not valid_relations.empty:
        section_map = df_sentences.set_index(['note_id', 'sentence_index'])['section_name'].to_dict()
        valid_relations['section_name'] = valid_relations.apply(
            lambda x: section_map.get((x['note_id'], x['sentence_index_1']), "unknown"), axis=1
        )

    if valid_sections and not valid_relations.empty:
        valid_relations = valid_relations[valid_relations['section_name'].isin(valid_sections)]
    return valid_relations.drop_duplicates(
        subset=['note_id', 'start_1', 'start_2']
    ).reset_index(drop=True)