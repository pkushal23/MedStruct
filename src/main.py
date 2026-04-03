from src.fetch_notes import fetch_one_note
from src.preprocess import clean_text
from src.ner_module import extract_medical_entities
from src.verify_module import get_prescriptions, verify_drugs
from src.export_module import save_csv, save_json

def run_pipeline():
    # 1. Fetch note
    df = fetch_one_note()
    row = df.iloc[0]

    subject_id = int(row["subject_id"])
    hadm_id = int(row["hadm_id"])
    note_id = row["note_id"]
    raw_text = row["text"]

    # 2. Clean text
    cleaned_text = clean_text(raw_text)

    # 3. Extract entities
    entities = extract_medical_entities(cleaned_text)

    # 4. Validate drugs
    prescriptions = get_prescriptions(hadm_id)
    verified_drugs = verify_drugs(entities, prescriptions)

    # 5. Build final output
    final_output = {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "note_id": note_id,
        "cleaned_note_preview": cleaned_text[:1000],
        "entities": entities,
        "verified_drugs": verified_drugs
    }

    # 6. Save files
    save_csv(entities, "data/processed/entities.csv")
    save_csv(verified_drugs, "data/processed/verified_drugs.csv")
    save_json(final_output, "data/processed/final_output.json")

    print("✅ Pipeline completed successfully.")
    print("Saved:")
    print("- data/processed/entities.csv")
    print("- data/processed/verified_drugs.csv")
    print("- data/processed/final_output.json")

if __name__ == "__main__":
    run_pipeline()