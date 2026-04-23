import pandas as pd


def calculate_iou(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    
    # Calculate intersection
    intersection = max(0, min(end1,end2) - max(start1, start2))
    
    # Calculate union
    union = max(end1, end2) - min(start1, start2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def merge_and_deduplicate(df_list):
    combined_df = pd.concat(df_list, ignore_index=True) if isinstance(df_list, list) else df_list

    if combined_df.empty:
        return combined_df

    # Normalize schema across NER modules (some use entity_group, others entity_type).
    if 'entity_type' not in combined_df.columns and 'entity_group' in combined_df.columns:
        combined_df = combined_df.copy()
        combined_df['entity_type'] = combined_df['entity_group']
    
    final_entities = []
    for (note_id, sent_idx), group in combined_df.groupby(['note_id', 'sentence_index']):
        accepted_entities = []
        for _, row in group.iterrows():
            current_span = (row['start'], row['end'])
            is_overlap = False
            
            for i, accepted in enumerate(accepted_entities):
                accepted_span = (accepted['start'], accepted['end'])
                iou = calculate_iou(current_span, accepted_span)
                
                if iou > 0.40:  # Threshold for considering as same entity
                    is_overlap = True
                    current_entity_type = row.get('entity_type', row.get('entity_group', ''))
                    accepted_entity_type = accepted.get('entity_type', accepted.get('entity_group', ''))

                    is_current_med7 = row['source_model'] == 'Med7' and current_entity_type == 'DRUG'
                    is_accepted_med7 = accepted['source_model'] == 'Med7' and accepted_entity_type == 'DRUG'
                    
                    if is_current_med7 and not is_accepted_med7:
                        accepted_entities[i] = row
                    elif not is_current_med7 and is_accepted_med7:
                        
                        if len(str(row['word'])) > len(str(accepted['word'])):
                            accepted_entities[i] = row
                    break
            if not is_overlap:
                accepted_entities.append(row)
        final_entities.extend(accepted_entities)
            
    return pd.DataFrame(final_entities)