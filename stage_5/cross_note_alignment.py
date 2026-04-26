import pandas as pd

class CrossNoteJoiner:
    def build_longitudinal_edges(self, df):
        """
        Aggregates relations across the entire patient admission.
        """
        # 1. Filter out NLP noise and invalid semantic mappings
        clean_df = df[
            (df['cui_1'] != 'UNMAPPED') & 
            (df['cui_2'] != 'UNMAPPED') &
            (df['target_semantic_valid'] == True)
        ].copy()
        
        # 2. Group by admission and the normalized CUI pair
        # We group by CUI, not text, to ensure 'afib' and 'Atrial Fibrillation' merge.
        grouped = clean_df.groupby(['hadm_id', 'cui_1', 'cui_2', 'relation_type'])

        longitudinal_edges = []
        
        for name, group in grouped:
            hadm_id, c1, c2, rel = name
            
            # Track unique notes and sections for the consensus scorer
            notes_found_in = group['note_id'].unique().tolist()
            sections_found = group['section_name'].unique().tolist()
            
            longitudinal_edges.append({
                'hadm_id': hadm_id,
                'source_cui': c1,
                'source_name': group['canonical_name_1'].iloc[0],
                'target_cui': c2,
                'target_name': group['canonical_name_2'].iloc[0],
                'relation': rel,
                'note_occurrences': notes_found_in,
                'sections_found': sections_found,
                'mention_count': len(group),
                'base_confidence': group['model_confidence'].mean()
            })

        return pd.DataFrame(longitudinal_edges)