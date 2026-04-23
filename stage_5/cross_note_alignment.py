import pandas as pd

class CrossNoteJoiner:
    def build_longitudinal_edges(self, df):
        # 1. Filter out the NLP noise
        clean_df = df[(df['cui_1'] != 'UNMAPPED') & (df['cui_2'] != 'UNMAPPED')].copy()
        clean_df = clean_df[clean_df['cui_1'] != clean_df['cui_2']]
        # 2. Group by admission and specific entity pairs
        grouped = clean_df.groupby(['hadm_id', 'cui_1', 'canonical_name_1', 'cui_2', 'canonical_name_2', 'relation_type'])

        longitudinal_edges = []
        
        # 3. Aggregate across notes
        for name, group in grouped:
            hadm_id, c1, name1, c2, name2, rel = name
            notes_found_in = group['note_id'].unique().tolist()
            
            longitudinal_edges.append({
                'hadm_id': hadm_id,
                'source_cui': c1,
                'source_name': name1,
                'target_cui': c2,
                'target_name': name2,
                'relation': rel,
                'note_occurrences': notes_found_in,
                'mention_count': len(group),
                'base_confidence': group['model_confidence'].mean()
            })

        return pd.DataFrame(longitudinal_edges)