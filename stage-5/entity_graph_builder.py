class EntityGraphBuilder:
    def build_json_graph(self, df_scored_edges):
        admissions_graph = {}

        # Build a distinct graph for each hospital admission
        for hadm_id, group in df_scored_edges.groupby('hadm_id'):
            nodes = {}
            edges = []

            for _, row in group.iterrows():
                # Register Source Node (Drug/Disease)
                if row['source_cui'] not in nodes:
                    nodes[row['source_cui']] = {"id": row['source_cui'], "name": row['source_name']}

                # Register Target Node (Drug/Disease)
                if row['target_cui'] not in nodes:
                    nodes[row['target_cui']] = {"id": row['target_cui'], "name": row['target_name']}

                # Create the Connecting Edge
                edges.append({
                    "source": row['source_cui'],
                    "target": row['target_cui'],
                    "relation": row['relation'],
                    "alignment_score": round(row['alignment_score'], 3),
                    "notes_referenced": row['note_occurrences']
                })

            admissions_graph[str(hadm_id)] = {
                "hadm_id": int(hadm_id),
                "nodes": list(nodes.values()),
                "edges": edges
            }

        return admissions_graph