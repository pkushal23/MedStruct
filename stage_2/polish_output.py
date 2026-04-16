import pandas as pd

def final_polish(df):
    """
    Final data scientist's cleanup:
    1. Removes persistent subword artifacts.
    2. Resolves overlapping spans by keeping the longest word.
    """

    df = df[~df['word'].str.contains('##', na=False)].copy()
    

    df['word_len'] = df['word'].str.len()
    df = df.sort_values(['note_id', 'sentence_index', 'word_len'], ascending=[True, True, False])
    
    final_rows = []
    for (note_id, sent_idx), group in df.groupby(['note_id', 'sentence_index']):
        occupied_positions = set()
        
        for _, row in group.iterrows():
            current_span = set(range(row['start'], row['end']))
            
            if not (current_span & occupied_positions):
                final_rows.append(row)
                occupied_positions.update(current_span)
                
    return pd.DataFrame(final_rows).drop(columns=['word_len'])


df_entities = pd.read_csv('../data/processed/entities.csv')
clean_entities = final_polish(df_entities)
clean_entities.to_csv('../data/processed/entities_refined.csv', index=False)