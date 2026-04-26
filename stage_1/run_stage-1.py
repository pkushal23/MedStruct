import os
import logging
from dotenv import load_dotenv

# Import pipeline modules
from loader import MimicLoader
from text_cleaner import clean_dataframe
from section_segmenter import segment_dataframe
from tokenizer import ClinicalTokenizer
from icd_mapper import map_icd_codes

# Setup Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    OUTPUT_DIR = "../data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize classes
    loader = MimicLoader(project_id=PROJECT_ID)
    tokenizer = ClinicalTokenizer()

    # --- PART A: PROCESS DISCHARGE SUMMARIES ---
    logging.info("=== Starting Discharge Summary Pipeline ===")
    # Increased limit to 500 for scaling
    df_ds_raw = loader.fetch_notes('discharge', limit=500) 
    df_ds_clean = clean_dataframe(df_ds_raw)
    df_ds_seg = segment_dataframe(df_ds_clean, 'discharge')
    
    
    # --- PART B: PROCESS RADIOLOGY REPORTS ---
    logging.info("=== Starting Radiology Report Pipeline ===")
    # Increased limit to 500 for scaling
    df_rr_raw = loader.fetch_notes('radiology', limit=500)
    df_rr_clean = clean_dataframe(df_rr_raw)
    df_rr_seg = segment_dataframe(df_rr_clean, 'radiology')
    
    # --- PART C: TOKENIZE BOTH ---
    import pandas as pd
    df_combined_seg = pd.concat([df_ds_seg, df_rr_seg], ignore_index=True)
    
    df_sections, df_sentences = tokenizer.tokenize_dataframe(df_combined_seg)
    
    # Save text outputs
    df_sections.to_csv(f"{OUTPUT_DIR}/sections.csv", index=False)
    df_sentences.to_csv(f"{OUTPUT_DIR}/sentences.csv", index=False)
    logging.info("Saved sections.csv and sentences.csv")

    # --- PART D: ICD HARMONIZATION ---
    logging.info("=== Starting ICD Harmonization ===")
    df_diag_raw = loader.fetch_diagnoses_icd()
    df_diag_mapped = map_icd_codes(df_diag_raw)
    df_diag_mapped.to_csv(f"{OUTPUT_DIR}/diagnoses_icd_mapped.csv", index=False)
    logging.info("Saved diagnoses_icd_mapped.csv")

    logging.info("🎉 STAGE 1 COMPLETE! Ready for Stage 2 (NER).")

if __name__ == "__main__":
    main()