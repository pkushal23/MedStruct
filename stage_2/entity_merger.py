import pandas as pd

BASE_OVERLAP_THRESHOLD = 0.40
MEDICATION_OVERLAP_THRESHOLD = 0.60

MEDICATION_LABELS = {
    'drug',
    'medication',
    'chemical',
}

LABEL_PRIORITY = {
    'drug': 300,
    'medication': 260,
    'route': 220,
    'strength': 210,
    'frequency': 200,
    'dose': 190,
    'dosage': 190,
    'chemical': 120,
    'treatment': 110,
    'problem': 100,
    'procedure': 90,
}

SOURCE_PRIORITY = {
    'med7': 300,
    'clinicalbert': 150,
    'radbert': 100,
}


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


def _normalize_label(row):
    label = row.get('entity_type', row.get('entity_group', ''))
    return str(label).strip().lower()


def _normalize_source(row):
    return str(row.get('source_model', '')).strip().lower()


def _is_medication_entity(row):
    return _normalize_label(row) in MEDICATION_LABELS or _normalize_source(row) == 'med7'


def _entity_priority(row):
    label_priority = LABEL_PRIORITY.get(_normalize_label(row), 0)
    source_priority = SOURCE_PRIORITY.get(_normalize_source(row), 0)
    span_length = int(row.get('end', 0)) - int(row.get('start', 0))
    score = float(row.get('score', 0.0))
    word_length = len(str(row.get('word', '')))
    return (source_priority + label_priority, span_length, score, word_length)


def _is_better_entity(candidate, incumbent):
    return _entity_priority(candidate) > _entity_priority(incumbent)


def _overlap_threshold(row_a, row_b):
    if _is_medication_entity(row_a) and _is_medication_entity(row_b):
        return MEDICATION_OVERLAP_THRESHOLD
    return BASE_OVERLAP_THRESHOLD

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
        group = group.copy()
        group['_priority'] = group.apply(_entity_priority, axis=1)
        group = group.sort_values(
            by=['_priority', 'start', 'end', 'score'],
            ascending=[False, True, True, False],
        )

        for _, row in group.iterrows():
            current_span = (row['start'], row['end'])
            is_overlap = False
            
            for i, accepted in enumerate(accepted_entities):
                accepted_span = (accepted['start'], accepted['end'])
                iou = calculate_iou(current_span, accepted_span)

                if iou > _overlap_threshold(row, accepted):
                    is_overlap = True
                    if _is_better_entity(row, accepted):
                        accepted_entities[i] = row
                    break
            if not is_overlap:
                accepted_entities.append(row)
        final_entities.extend(accepted_entities)
            
    final_df = pd.DataFrame(final_entities)
    if '_priority' in final_df.columns:
        final_df = final_df.drop(columns=['_priority'])
    return final_df