import json

class EntityGraphBuilder:
    def build_json_graph(self, df_scored_edges):
        """
        Converts clinical edges into an analytical Graph JSON format.
        Integrates 'Explainable Edge' logic by flagging consensus based on 
        note frequency and section diversity.
        """
        admissions_graph = {}

        for hadm_id, group in df_scored_edges.groupby('hadm_id'):
            nodes = {}
            edges = []

            for _, row in group.iterrows():
                # 1. Node Registration with Centrality Analytics
                # Centrality tracks how many relationships are connected to this node
                for role in ['source', 'target']:
                    cui = row[f'{role}_cui']
                    if cui not in nodes:
                        nodes[cui] = {
                            "id": cui, 
                            "name": row[f'{role}_name'],
                            "centrality": 0 
                        }
                    nodes[cui]["centrality"] += 1 

                # 2. Edge Creation with Consensus Alignment
                # A relationship is flagged as 'consensus' if it has been verified
                # across multiple notes or multiple clinical sections.
                sections = row.get('sections_found', [])
                notes = row.get('note_occurrences', [])
                
                is_consensus = len(notes) > 1 or len(sections) > 1

                edges.append({
                    "source": row['source_cui'],
                    "target": row['target_cui'],
                    "relation": row['relation'],
                    "alignment_score": round(row['alignment_score'], 3),
                    "evidence_notes": notes,
                    "evidence_sections": sections,
                    "is_consensus": is_consensus
                })

            # 3. Admission Summary Statistics for the Report
            admissions_graph[str(hadm_id)] = {
                "hadm_id": int(hadm_id),
                "summary": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "consensus_ratio": sum(1 for e in edges if e['is_consensus']) / len(edges) if edges else 0
                },
                "nodes": list(nodes.values()),
                "edges": edges
            }

        return admissions_graph