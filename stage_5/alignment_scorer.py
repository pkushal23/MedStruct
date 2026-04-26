import pandas as pd

class AlignmentScorer:
    def __init__(self, multi_note_boost=0.08, cross_section_bonus=0.05):
        # Increased boost for multi-note consensus
        self.multi_note_boost = multi_note_boost
        self.cross_section_bonus = cross_section_bonus

    def score_edges(self, df_edges):
        """
        Scores edges based on cross-note consistency and evidence diversity.
        """
        scored_df = df_edges.copy()

        def calculate_alignment(row):
            base = row['base_confidence']
            unique_notes = len(row['note_occurrences'])
            
            # 1. Multi-Note Consistency Boost
            boost = (unique_notes - 1) * self.multi_note_boost if unique_notes > 1 else 0
            
            # 2. Section Diversity Bonus (Consensus between different clinical views)
            # If the relation is found in both 'HPI' and 'Hospital Course', it's highly likely to be true.
            sections = set(row.get('sections_found', []))
            diversity_bonus = self.cross_section_bonus if len(sections) > 1 else 0
            
            final_score = min(1.0, base + boost + diversity_bonus)
            return final_score

        scored_df['alignment_score'] = scored_df.apply(calculate_alignment, axis=1)
        return scored_df