import pandas as pd
from tqdm import tqdm

# Enable tqdm with pandas apply
tqdm.pandas(desc="Mapping to UMLS")

# Semantic type constraints based on entity type.
# scispaCy exposes semantic types as TUIs, so we translate them to canonical
# names before comparing against these labels.
SEMANTIC_TYPE_CONSTRAINTS = {
    'drug': {'Pharmacologic Substance', 'Clinical Drug'},
    'medication': {'Pharmacologic Substance', 'Clinical Drug'},
    'chemical': {'Pharmacologic Substance', 'Clinical Drug', 'Organic Chemical'},
    'problem': {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning'},
    'test': {'Diagnostic Procedure', 'Laboratory Procedure'},
    'treatment': {'Therapeutic or Preventive Procedure', 'Clinical Drug', 'Pharmacologic Substance'},
}

# Rejection patterns for semantic validation to prevent illogical mappings (e.g., treating a body part)
REJECTED_SEMANTIC_TYPES = {'Body Part', 'Body Location', 'Anatomical Structure'}

class CUIMapper:
    def __init__(self, linker):
        self.linker = linker
        self.semantic_type_tree = linker.linker.kb.semantic_type_tree

    def _normalize_entity_group(self, entity_group):
        return str(entity_group).strip().lower()

    def _get_semantic_type(self, cui):
        """Retrieve canonical semantic type names for a given CUI."""
        try:
            entity = self.linker.linker.kb.cui_to_entity.get(cui)
            if entity:
                semantic_types = []
                for tui in getattr(entity, 'types', []):
                    try:
                        semantic_types.append(self.semantic_type_tree.get_canonical_name(tui))
                    except Exception:
                        semantic_types.append(tui)
                return semantic_types
            return []
        except Exception:
            return []

    def _validate_semantic_type(self, cui, entity_group, is_target=False):
        """Validate if CUI's semantic type matches entity_group constraints."""
        if cui == "UNMAPPED":
            return False
        
        entity_group = self._normalize_entity_group(entity_group)
        semantic_types = self._get_semantic_type(cui)
        
        # Check if semantic type is in rejection list
        if any(rejected in semantic_types for rejected in REJECTED_SEMANTIC_TYPES):
            return False
        
        # If no constraints defined for this entity_group, accept any valid mapping
        if entity_group not in SEMANTIC_TYPE_CONSTRAINTS:
            return cui != "UNMAPPED"
        
        # Check if semantic type matches constraints for this entity_group
        allowed_types = SEMANTIC_TYPE_CONSTRAINTS[entity_group]
        return any(st in allowed_types for st in semantic_types) or len(semantic_types) == 0

    def _adjust_candidate_score(self, base_score, semantic_types, entity_group, relation_context=None):
        """
        Bias candidate scores using relation context and semantic type compatibility.
        Implements 'Relational Context Bias' to prioritize clinically logical CUIs.
        """
        adjusted_score = float(base_score)
        context = relation_context or {}
        relation_type = str(context.get('relation_type', '')).strip().upper()
        role = str(context.get('role', '')).strip().lower()
        entity_group = self._normalize_entity_group(entity_group)

        # Logic for TREATS: Target should be a Problem/Finding
        if relation_type == 'TREATS' and role == 'target':
            allowed_target_types = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning'}
            if any(st in allowed_target_types for st in semantic_types):
                adjusted_score += 0.15  # Increased bias for valid targets
            else:
                adjusted_score -= 0.25  # Stronger penalty for anatomical or procedural targets
        
        # Logic for CAUSES (ADEs): Target should be a Problem or Adverse Event
        elif relation_type == 'CAUSES' and role == 'target':
            allowed_target_types = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning', 'Adverse Event'}
            if any(st in allowed_target_types for st in semantic_types):
                adjusted_score += 0.10
            else:
                adjusted_score -= 0.20

        # Preference for Pharmacologic Substances in drug/medication roles
        if entity_group in {'drug', 'medication', 'chemical'}:
            if any(st in {'Pharmacologic Substance', 'Clinical Drug'} for st in semantic_types):
                adjusted_score += 0.05
            else:
                adjusted_score -= 0.10

        # Universal penalty for strictly anatomical terms in relation slots
        if any(rejected in semantic_types for rejected in REJECTED_SEMANTIC_TYPES):
            adjusted_score -= 0.35

        return adjusted_score

    def _get_top_k_cui(self, text, entity_group, k=3, relation_context=None):
        """Get top K candidate CUIs with semantic type filtering and contextual re-ranking."""
        text = str(text).lower().strip()
        doc = self.linker.nlp(text)
        entity_group = self._normalize_entity_group(entity_group)
        
        candidates = []
        
        # Collect all candidates from all entities identified in the string
        for ent in doc.ents:
            if ent._.kb_ents:
                for cui, score in ent._.kb_ents[:k]:
                    try:
                        concept = self.linker.linker.kb.cui_to_entity[cui]
                        semantic_types = self._get_semantic_type(cui)
                        
                        # Apply contextual re-ranking bias
                        adjusted_score = self._adjust_candidate_score(
                            score,
                            semantic_types,
                            entity_group,
                            relation_context=relation_context,
                        )

                        # Only add to candidates if it passes the base semantic gate
                        if self._validate_semantic_type(cui, entity_group, is_target=False):
                            candidates.append((cui, concept.canonical_name, adjusted_score))
                    except KeyError:
                        pass
        
        # Sort by adjusted score and return top candidate
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[0]
            return best[0], best[1]
        else:
            return "UNMAPPED", "UNMAPPED"

    def _map_row(self, row):
        """Map both source and target entities with role-based semantic validation."""
        # Detect group columns regardless of stage suffixes
        entity_1_group = row.get('entity_type_1', row.get('entity_1_type', 'problem'))
        entity_2_group = row.get('entity_type_2', row.get('entity_2_type', 'problem'))
        relation_type = row.get('relation_type', '')
        
        # Map the Source Entity (role: source)
        cui_1, canonical_1 = self._get_top_k_cui(
            row['word_1'],
            entity_1_group,
            k=3,
            relation_context={'relation_type': relation_type, 'role': 'source'},
        )
        
        # Map the Target Entity (role: target)
        cui_2, canonical_2 = self._get_top_k_cui(
            row['word_2'],
            entity_2_group,
            k=3,
            relation_context={'relation_type': relation_type, 'role': 'target'},
        )
        
        return pd.Series([cui_1, canonical_1, cui_2, canonical_2])

    def map_dataframe(self, df):
        """Map entire dataframe with top-K ranking and role-based re-ranking."""
        df_mapped = df.copy()
        
        # Apply the mapping function to generate high-resolution CUI columns
        df_mapped[['cui_1', 'canonical_name_1', 'cui_2', 'canonical_name_2']] = df_mapped.progress_apply(self._map_row, axis=1)
            
        return df_mapped