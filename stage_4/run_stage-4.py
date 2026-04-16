import os
import pandas as pd
from umls_setup import check_dependencies
from ontology_validation import OntologyLinker
from cui_mapper import CUIMapper

def main():
    # 1. Run the safety check first
    if not check_dependencies():
        return  # Exit if model is not installed

    # 2. Setup paths
    input_file = "../data/processed/relations_verified.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading verified relations from Stage 3...")
    df_relations = pd.read_csv(input_file)

    # 3. Initialize components
    linker = OntologyLinker()
    mapper = CUIMapper(linker)

    # 4. Process the data
    df_mapped = mapper.map_dataframe(df_relations)

    # 5. Save the output
    output_file = os.path.join(output_dir, "relations_normalized.csv")
    df_mapped.to_csv(output_file, index=False)
    
    print(f"\nStage 4 Complete! Normalized {len(df_mapped)} relationships.")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    main()