import os
import pandas as pd
from relation_extractor import RelationExtractor

def main():
    input_entities = "../data/processed/entities_refined.csv"
    input_sentences = "../data/processed/sentences.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading  refined Stage 2 data...")
    df_entities = pd.read_csv(input_entities)
    df_sentences = pd.read_csv(input_sentences)

    extractor = RelationExtractor(
        drug_labels=['DRUG'], 
        disease_labels=['problem'],
        device=0
    )
    
    ade_sections = ['HPI', 'Hospital Course', 'Assessment/Plan', 'Findings', 'Impression']
    
    print(f"Extracting relations from {len(ade_sections)} targetd clinical sections...")

    df_relations = extractor.extract_drug_disease(df_entities=df_entities, df_sentences= df_sentences, window=1, valid_sections=ade_sections, threshold=0.50)

    output_file = os.path.join(output_dir, "relations_verified.csv")
    
    if not df_relations.empty:
        df_relations = df_relations.sort_values(by='model_confidence', ascending=False)
        df_relations.to_csv(output_file, index=False)
        print(f"Stage 3 Cmplete! Verified {len(df_relations)} semantic relationships.")
        
        ade_count = len(df_relations[df_relations['relation_type'] == 'CAUSES'])
        print(f"--> Found {ade_count} potential ADE relationships (CAUSES) for Stage 4.")
    else:
        print(f"Stage 3 Complete! No high-confidence relationships found.")
        

if __name__ == "__main__":
    main()