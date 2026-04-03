from google.cloud import bigquery

def get_bq_client():
    return bigquery.Client(project="gen-lang-client-0941786636")

