import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_gem_crosswalk(gem_csv_path: str) -> dict:
    """
    Loads the CMS GEM crosswalk file into a dictionary mapping ICD-9 to ICD-10.
    Expected CSV columns: ['icd9_code', 'icd10_code']
    """
    try:
        df_gem = pd.read_csv(gem_csv_path, dtype=str)
        # Create a dictionary mapping ICD-9 directly to ICD-10
        return dict(zip(df_gem['icd9_code'], df_gem['icd10_code']))
    except FileNotFoundError:
        logger.warning(f"GEM crosswalk file not found at {gem_csv_path}. Mapping will be skipped.")
        return {}

def map_icd_codes(df_diagnoses: pd.DataFrame, gem_csv_path: str = '../data/external/gem_icd9_to_icd10.csv') -> pd.DataFrame:
    """
    Converts ICD-9 codes to ICD-10. Flags unmapped codes.
    """
    logger.info("Starting ICD-9 to ICD-10 mapping...")
    gem_dict = load_gem_crosswalk(gem_csv_path)
    
    df_mapped = df_diagnoses.copy()
    
    def apply_mapping(row):
        # If it's already ICD-10, leave it alone
        if str(row['icd_version']) == '10':
            return row['icd_code'], '10', False
            
        # If it's ICD-9, try to look it up
        if str(row['icd_version']) == '9':
            icd9 = str(row['icd_code']).strip()
            if icd9 in gem_dict:
                return gem_dict[icd9], '10', False # Successfully mapped
            else:
                return icd9, '9', True # Unmapped flag
                
        return row['icd_code'], row['icd_version'], True

    # Apply the mapping logic
    mapped_results = df_mapped.apply(apply_mapping, axis=1, result_type='expand')
    df_mapped[['icd_code', 'icd_version', 'is_unmapped']] = mapped_results
    
    unmapped_count = df_mapped['is_unmapped'].sum()
    logger.info(f"Mapping complete. {unmapped_count} codes could not be mapped to ICD-10.")
    
    return df_mapped