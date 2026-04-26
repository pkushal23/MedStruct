import os
import pandas as pd
from tqdm import tqdm
# Import local pipeline components
from umls_setup import check_dependencies
from ontology_validation import OntologyLinker
from cui_mapper import CUIMapper

# Semantic type validation constants for medical logic
VALID_TREATS_TARGETS = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning'}
VALID_CAUSES_TARGETS = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning', 'Adverse Event'}
REJECTED_TARGET_TYPES = {'Body Part', 'Body Location', 'Anatomical Structure'}

def _validate_target_semantic_type(linker, cui, relation_type):
    """Enforces clinical logic on final mappings."""
    if cui == "UNMAPPED":
        return False
    try:
        entity = linker.linker.kb.cui_to_entity.get(cui)
        if not entity: return False
        semantic_types = [linker.linker.kb.semantic_type_tree.get_canonical_name(t) for t in entity.types]
        if any(rejected in semantic_types for rejected in REJECTED_TARGET_TYPES):
            return False
        valid_set = VALID_TREATS_TARGETS if relation_type == 'TREATS' else VALID_CAUSES_TARGETS
        return any(st in valid_set for st in semantic_types) or len(semantic_types) == 0
    except:
        return False

def process_track(df_track, mapper, relation_type, threshold, output_file):
    """Processes a track from extraction to validation."""
    print(f"\n--- Processing {relation_type} Track ---")
    df_f = df_track[(df_track['relation_type'] == relation_type) & (df_track['model_confidence'] >= threshold)].copy()
    if df_f.empty: return None
    
    df_m = mapper.map_dataframe(df_f)

    # Final semantic validation after deterministic mapping
    df_m['target_semantic_valid'] = df_m.apply(lambda r: _validate_target_semantic_type(mapper.linker, r['cui_2'], relation_type), axis=1)
    df_m.to_csv(output_file, index=False)
    return df_m

def main():
    if not check_dependencies(): return #
    
    input_file = "../data/processed/relations_verified.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    df_raw = pd.read_csv(input_file)
    
    linker = OntologyLinker() #
    mapper = CUIMapper(linker) #
    df_ade = process_track(df_raw, mapper, 'CAUSES', 0.75, os.path.join(output_dir, "relations_normalized_ade.csv"))
    df_treat = process_track(df_raw, mapper, 'TREATS', 0.65, os.path.join(output_dir, "relations_normalized_treatment.csv"))
    
    # Consolidate results
    df_final = pd.concat([d for d in [df_ade, df_treat] if d is not None], ignore_index=True)
    df_final.to_csv(os.path.join(output_dir, "relations_normalized.csv"), index=False)
    
    mapped_count = (df_final['cui_1'] != 'UNMAPPED').sum() + (df_final['cui_2'] != 'UNMAPPED').sum()
    total_slots = len(df_final) * 2
    print(f"\nStage 4 Complete. Final coverage: {(mapped_count / total_slots):.1%}")

if __name__ == "__main__":
    main()
    