import os
import pandas as pd
import json
from cross_note_alignment import CrossNoteJoiner
from alignment_scorer import AlignmentScorer
from entity_graph_builder import EntityGraphBuilder

def main():
    input_file = "../data/processed/relations_normalized.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading normalized relations from Stage 4...")
    df_normalized = pd.read_csv(input_file)

    # 1. Join relationships across time/notes
    print("Joining entities across longitudinal notes...")
    joiner = CrossNoteJoiner()
    df_joined = joiner.build_longitudinal_edges(df_normalized)

    # 2. Score the longitudinal alignment
    print("Scoring cross-note clinical alignment...")
    scorer = AlignmentScorer(multi_note_boost=0.05)
    df_scored = scorer.score_edges(df_joined)

    # 3. Build the strict graph payload
    print("Building entity knowledge graphs...")
    builder = EntityGraphBuilder()
    graph_payload = builder.build_json_graph(df_scored)

    # 4. Save Artifacts
    json_output = os.path.join(output_dir, "longitudinal_graphs.json")
    with open(json_output, 'w') as f:
        json.dump(graph_payload, f, indent=4)
        
    csv_output = os.path.join(output_dir, "longitudinal_edges.csv")
    df_scored.to_csv(csv_output, index=False)

    print(f"\nStage 5 Complete! Created knowledge graphs for {len(graph_payload)} admissions.")
    print(f"Graph JSON saved to: {json_output}")
    print(f"Edge CSV saved to: {csv_output}")

if __name__ == "__main__":
    main()