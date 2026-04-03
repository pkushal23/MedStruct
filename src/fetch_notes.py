from src.bigquery_client import get_bq_client

def fetch_one_note():
    client = get_bq_client()

    query = """
    SELECT 
        subject_id,
        hadm_id,
        note_id,
        text
    FROM `physionet-data.mimiciv_note.discharge`
    WHERE text IS NOT NULL
    LIMIT 1
    """

    df = client.query(query).to_dataframe()
    return df