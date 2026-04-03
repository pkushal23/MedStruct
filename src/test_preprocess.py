from src.fetch_notes import fetch_one_note
from src.preprocess import clean_text

df = fetch_one_note()

raw_text = df.iloc[0]["text"]
cleaned_text = clean_text(raw_text)

print("=== CLEANED NOTE ===\n")
print(cleaned_text[:2000])