from src.fetch_notes import fetch_one_note

df = fetch_one_note()

print("=== DataFrame Preview ===")
print(df.head())

print("\n=== Sample Note Text ===\n")
print(df.iloc[0]["text"][:1500])