import pandas as pd


def merge_and_deduplicate(df_list):
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.sort_values(by=['note_id', 'score'], ascending=[True, False])
    
    final_entities = []
    for (note_id, sent_idx), group in combined_df.groupby(['note_id', 'sentence_index']):
        seen_spans = set()
        for _, row in group.iterrows():
            span = (row['start'], row['end'])
            if span not in seen_spans:
                final_entities.append(row)
                seen_spans.add(span)
            
    return pd.DataFrame(final_entities)

def stitch_subwords(df):
    if df.empty:
        return df
    
    # Ensure entities are sorted by appearance
    df = df.sort_values(['note_id', 'sentence_index', 'start'])
    final_entities = []
    
    for _, group in df.groupby(['note_id', 'sentence_index']):
        current_entity = None
        
        for _, row in group.iterrows():
            if current_entity is None:
                current_entity = row.to_dict()
                continue
            
            # Check if current word is a subword and physically touches the previous word
            if str(row['word']).startswith('##') and row['start'] == current_entity['end']:
                # Stitch word (remove ##) and update the end boundary
                current_entity['word'] += row['word'][2:]
                current_entity['end'] = row['end']
                # Keep the higher score of the two
                current_entity['score'] = max(current_entity['score'], row['score'])
            else:
                # No stitch possible; move the current entity to final list
                final_entities.append(current_entity)
                current_entity = row.to_dict()
        
        if current_entity:
            final_entities.append(current_entity)
            
    return pd.DataFrame(final_entities)