import os
import pandas as pd
from relation_extractor import RelationExtractor

def main():
    input_entities = "../data/processed/entities_refined.csv"
    input_sentences = "../data/processed/sentences.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df_entities = pd.read_csv(input_entities).head(2000)
    df_sentences = pd.read_csv(input_sentences)

    extractor = RelationExtractor(
        drug_labels=['DRUG'], 
        disease_labels=['problem'],
        device=-1
    )

    df_relations = extractor.extract_drug_disease(df_entities, df_sentences, window=1, threshold=0.7)

    output_file = os.path.join(output_dir, "relations_verified.csv")
    df_relations.to_csv(output_file, index=False)
    
    print(f"Stage 3 Complete! Verified {len(df_relations)} semantic relationships.")

if __name__ == "__main__":
    main()