import pandas as pd

class AlignmentScorer:
    def __init__(self, multi_note_boost=0.05):
        # How much to boost the score for each additional note it appears in
        self.multi_note_boost = multi_note_boost

    def score_edges(self, df_edges):
        scored_df = df_edges.copy()

        def calculate_alignment(row):
            base = row['base_confidence']
            unique_notes_count = len(row['note_occurrences'])
            
            # Boost score if mentioned in multiple distinct notes over time
            if unique_notes_count > 1:
                boost = (unique_notes_count - 1) * self.multi_note_boost
                final_score = min(1.0, base + boost) # Cap at 100% confidence
            else:
                final_score = base
                
            return final_score

        scored_df['alignment_score'] = scored_df.apply(calculate_alignment, axis=1)
        return scored_df