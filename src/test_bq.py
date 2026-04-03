from src.bigquery_client import get_bq_client

client = get_bq_client()

query = "SELECT 1 AS test"
df = client.query(query).to_dataframe()

print(df)