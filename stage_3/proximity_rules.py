import pandas as pd

def get_cooccurring_entities(df_entities, target_types, max_sentence_window):
    # Filter only for the entity types we care about
    df_filtered = df_entities[df_entities['entity_group'].isin(target_types)].copy()
    
    # Merge the dataframe with itself to create pairs of entities
    merged = pd.merge(
        df_filtered, 
        df_filtered, 
        on=['note_id', 'hadm_id'], 
        suffixes=('_1', '_2')
    )
    
    # THE FIX: Enforce a strict ordering so A->B is kept, but B->A is dropped
    merged = merged[merged['start_1'] < merged['start_2']]
    
    # Calculate how many sentences apart they are
    merged['sentence_diff'] = (merged['sentence_index_1'] - merged['sentence_index_2']).abs()
    
    # Keep only the pairs that fall within our window
    valid_relations = merged[merged['sentence_diff'] <= max_sentence_window]
    
    return valid_relations.drop_duplicates(
        subset=['note_id', 'start_1', 'start_2']
    ).reset_index(drop=True)