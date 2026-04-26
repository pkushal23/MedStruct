import pandas as pd
import logging
from google.cloud import bigquery

logger = logging.getLogger(__name__)

class MimicLoader:
    def __init__(self, project_id: str):
        self.client = bigquery.Client(project=project_id)

    def fetch_notes(self, table_type: str, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
        """
        Loads notes from either 'discharge' or 'radiology'.
        Filters out null hadm_ids as per Step 1.1 specs.
        """
        if table_type not in ['discharge', 'radiology']:
            raise ValueError("table_type must be 'discharge' or 'radiology'")

        query = f"""
        SELECT subject_id, hadm_id, note_id, text
        FROM `physionet-data.mimiciv_note.{table_type}`
        WHERE subject_id IN (
            SELECT subject_id FROM `physionet-data.mimiciv_note.{table_type}`
            WHERE hadm_id IS NOT NULL
            GROUP BY subject_id
            HAVING COUNT(note_id) > 1
            LIMIT 200  -- Targets 200 patients who have at least 2 notes each
        )
        AND hadm_id IS NOT NULL
        LIMIT {limit}
        """
        
        logger.info(f"Loading {table_type} notes (Limit: {limit}, Offset: {offset})...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Successfully loaded {len(df)} {table_type} notes.")
        return df

    def fetch_diagnoses_icd(self) -> pd.DataFrame:
        query = """
        SELECT subject_id, hadm_id, seq_num, icd_code, icd_version
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
        WHERE hadm_id IS NOT NULL
        LIMIT 5000 
        """
        logger.info("Loading 5000 real ICD codes for the prototype...")
        return self.client.query(query).to_dataframe()