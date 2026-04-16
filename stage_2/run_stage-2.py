import os
import pandas as pd
from clinical_ner import ClinicalNER
from med7_ner import Med7NER
from radbert_ner import RadBERTNER
from entity_merger import merge_and_deduplicate, stitch_subwords

def main():
    input_file = "../data/processed/sentences.csv"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading sentences...")
    df_sentences = pd.read_csv(input_file)

    rad_sections = ['Indication', 'Findings', 'Impression']
    df_ds = df_sentences[~df_sentences['section_name'].isin(rad_sections)]
    df_rr = df_sentences[df_sentences['section_name'].isin(rad_sections)]

    clinical_model = ClinicalNER()
    med7_model = Med7NER()
    radbert_model = RadBERTNER()

    print("Running NER Ensemble...")
    df_clinical_entities = clinical_model.process_dataframe(df_ds)
    df_med7_entities = med7_model.process_dataframe(df_ds)
    df_radbert_entities = radbert_model.process_dataframe(df_rr)


    df_radbert_entities = df_radbert_entities[~df_radbert_entities['entity_group'].str.contains('LABEL')]
    

    print("Stitching subword fragments...")
    all_raw_entities = pd.concat([
        df_clinical_entities, 
        df_med7_entities, 
        df_radbert_entities
    ], ignore_index=True)

    stitched_entities = stitch_subwords(all_raw_entities)


    print("Merging and deduplicating entities...")
    merged_entities = merge_and_deduplicate([stitched_entities])

    output_file = os.path.join(output_dir, "entities.csv")
    merged_entities.to_csv(output_file, index=False)
    
    print(f"Stage 2 Complete! Cleaned entities saved to: {output_file}")

if __name__ == "__main__":
    main()