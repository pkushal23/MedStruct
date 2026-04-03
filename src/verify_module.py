from src.bigquery_client import get_bq_client

def get_prescriptions(hadm_id):
    client = get_bq_client()

    query = f"""
    SELECT DISTINCT drug
    FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
    WHERE hadm_id = {int(hadm_id)}
      AND drug IS NOT NULL
    """

    df = client.query(query).to_dataframe()
    return df["drug"].dropna().tolist()


DRUG_SYNONYMS = {
    "ativan": "lorazepam",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen"
}

NOISE_WORDS = {
    "pt", "nad", "ctab", "q8h", "qhs", "u-100",
    "greasy", "fen", "hsq", "pcp", "sodium"
}

INVALID_PATTERNS = [
    "/",
]

def normalize_drug(drug):
    drug = drug.lower().strip()
    return DRUG_SYNONYMS.get(drug, drug)

def is_noise(drug):
    drug_lower = drug.lower().strip()

    if drug_lower in NOISE_WORDS:
        return True

    if len(drug_lower) < 4:
        return True

    for pattern in INVALID_PATTERNS:
        if pattern in drug_lower:
            return True

    return False

def verify_drugs(entities, prescriptions):
    prescriptions_lower = [p.lower().strip() for p in prescriptions]

    results = []

    for e in entities:
        if e["label"] != "CHEMICAL":
            continue

        drug = e["entity"].strip()

        if is_noise(drug):
            continue

        normalized = normalize_drug(drug)

        matched = any(
            normalized in p or p in normalized
            for p in prescriptions_lower
        )

        results.append({
            "drug": drug,
            "normalized": normalized,
            "status": "Verified" if matched else "Unverified"
        })

    return results