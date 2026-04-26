from fastapi import FastAPI, UploadFile, File, Form
import fitz
import pandas as pd
import numpy as np
import re
import sys
import os
import unicodedata
os.environ["transformers_safe_load"] = "1" 
os.environ["TRUST_REMOTE_CODE"] = "True"
from stage_1.text_cleaner import clean_dataframe
from stage_1.section_segmenter import segment_dataframe
from stage_1.tokenizer import ClinicalTokenizer
from stage_2.clinical_ner import ClinicalNER
from stage_2.med7_ner import Med7NER
from stage_2.radbert_ner import RadBERTNER
from stage_2.entity_merger import merge_and_deduplicate
from stage_2.polish_output import final_polish
from stage_2.radiology_enhancer import RadiologyEnhancer
from stage_3.relation_extractor import RelationExtractor
from stage_4.cui_mapper import CUIMapper
from stage_4.ontology_validation import OntologyLinker
from stage_5.cross_note_alignment import CrossNoteJoiner
from stage_5.alignment_scorer import AlignmentScorer
from stage_5.entity_graph_builder import EntityGraphBuilder

app = FastAPI()

# --- INITIALIZE PIPELINE COMPONENTS ---
tokenizer = ClinicalTokenizer()
clinical_model = ClinicalNER()
med7_model = Med7NER()
radbert_model = RadBERTNER()
rad_enhancer = RadiologyEnhancer()

# FIX 1: Add Medications to sections and normalize case for NER labels
ACTIVE_SECTIONS = ['HPI', 'Hospital Course', 'Assessment/Plan', 'Findings', 'Impression', 'Medications']
VALID_TREATS_TARGETS = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning'}
VALID_CAUSES_TARGETS = {'Disease or Syndrome', 'Finding', 'Sign or Symptom', 'Injury or Poisoning', 'Adverse Event'}
REJECTED_TARGET_TYPES = {'Body Part', 'Body Location', 'Anatomical Structure'}

rel_extractor = RelationExtractor(
    drug_labels=['DRUG', 'treatment', 'TREATMENT'], 
    disease_labels=['problem', 'PROBLEM'], 
    device=0 
)

linker = OntologyLinker()
mapper = CUIMapper(linker)
joiner = CrossNoteJoiner()
scorer = AlignmentScorer(multi_note_boost=0.05)
graph_builder = EntityGraphBuilder()


def _to_builtin_types(value):
    """Recursively convert NumPy/Pandas scalar values to JSON-safe Python builtins."""
    if isinstance(value, dict):
        return {str(_to_builtin_types(k)): _to_builtin_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin_types(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin_types(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _normalize_text_for_ocr_noise(text: str) -> str:
    """Normalize common PDF/OCR artifacts before sectioning and tokenization."""
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00ad", "")
    normalized = re.sub(r"\s*[-‐‑‒–—]\s*", " - ", normalized)
    # CRITICAL: Only collapse spaces/tabs, NOT newlines (needed for section detection)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _build_radiology_edges(
    relations: list,
    hadm_id: int,
    note_id: str,
    subject_id: int,
    severity_map: dict,
    confidence_threshold: float = 0.50,
) -> list:
    """
    Create graph edges from radiology finding relationships with confidence filtering.
    
    Args:
        relations: List of finding relationship dicts.
        hadm_id: Hospital admission ID.
        note_id: Report file name.
        subject_id: Patient subject ID.
        severity_map: Dict mapping entity names to severity levels.
        confidence_threshold: Min confidence (0-1) to include edge (default 0.50).
    
    Returns:
        List of edge dicts with alignment scores and severity annotations.
    """
    edge_rows = []
    for rel in relations:
        source_raw = str(rel.get("entity_1", "")).strip()
        target_raw = str(rel.get("entity_2", "")).strip()
        if not source_raw or not target_raw:
            continue

        conf = float(rel.get("confidence", 0.0) or 0.0)
        
        # Filter low-confidence relations
        if conf < confidence_threshold:
            continue

        source = source_raw.lower()
        target = target_raw.lower()
        source_sev = severity_map.get(source) or severity_map.get(source_raw)
        target_sev = severity_map.get(target) or severity_map.get(target_raw)

        edge_rows.append({
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "source_note": note_id,
            "entity_1": source,
            "entity_2": target,
            "relation_type": str(rel.get("relation_type", "RELATED_TO")),
            "model_confidence": conf,
            "source_type": rel.get("entity_type_1", "unknown"),
            "target_type": rel.get("entity_type_2", "unknown"),
            "source_severity": source_sev,
            "target_severity": target_sev,
            "alignment_score": round(0.6 + 0.35 * conf, 4),
        })

    return edge_rows


def _validate_target_semantic_type(linker, cui, relation_type):
    if cui == "UNMAPPED":
        return False

    try:
        entity = linker.linker.kb.cui_to_entity.get(cui)
        if not entity:
            return False

        semantic_types = [linker.linker.kb.semantic_type_tree.get_canonical_name(t) for t in entity.types]
        if any(rejected in semantic_types for rejected in REJECTED_TARGET_TYPES):
            return False

        valid_set = VALID_TREATS_TARGETS if relation_type == 'TREATS' else VALID_CAUSES_TARGETS
        return any(st in valid_set for st in semantic_types) or len(semantic_types) == 0
    except Exception:
        return False

@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    subject_id: int = Form(...),
    hadm_id: int = Form(...),
    note_type: str = Form(...)
):
    
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    print(f"[DEBUG] Raw PDF text length: {len(text)} chars")
    print(f"[DEBUG] First 500 chars: {text[:500]}")
    
    text = _normalize_text_for_ocr_noise(text)
    print(f"[DEBUG] After normalization: {len(text)} chars")

    df_raw = pd.DataFrame([{
        "subject_id": subject_id, 
        "hadm_id": hadm_id, 
        "note_id": file.filename, 
        "text": text
    }])

    # FIX 2: Map shorthand types to full words expected by the segmenter
    # Handle both short codes (DS, RR) and full names from UI (Discharge Summary, Radiology Report)
    type_map = {
        "DS": "discharge", 
        "RR": "radiology", 
        "discharge": "discharge", 
        "radiology": "radiology",
        "Discharge Summary": "discharge",
        "Radiology Report": "radiology"
    }
    mapped_type = type_map.get(note_type, "discharge")
    print(f"[DEBUG] Input note_type: '{note_type}' -> Mapped to: '{mapped_type}'")

    # --- STAGE 1: Text Processing ---
    df_clean = clean_dataframe(df_raw)
    print(f"[DEBUG] After clean_dataframe: {len(df_clean)} rows")
    if not df_clean.empty:
        print(f"[DEBUG] Clean text sample: {df_clean.iloc[0]['text'][:300] if 'text' in df_clean.columns else 'NO TEXT'}")
    
    df_seg = segment_dataframe(df_clean, mapped_type) # CRITICAL: Use mapped_type
    print(f"[DEBUG] After segment_dataframe: {len(df_seg)} rows")
    
    df_sections, df_sentences = tokenizer.tokenize_dataframe(df_seg)
    print(f"[DEBUG] After tokenize: {len(df_sections)} sections, {len(df_sentences)} sentences")

    # --- STAGE 2: NER ---
    # Note: RadBERT model checkpoint lacks pre-trained NER classifier (missing classifier.weight/bias)
    # so it returns generic LABEL_* tags. Using clinical NER ensemble for all document types instead.
    df_clin = clinical_model.process_dataframe(df_sentences)
    print(f"[DEBUG] After ClinicalNER: {len(df_clin)} entities")
    if not df_clin.empty:
        print(f"[DEBUG] ClinicalNER columns: {df_clin.columns.tolist()}")
        sample_cols = ['word', 'entity_group'] if 'entity_group' in df_clin.columns else df_clin.columns.tolist()[:3]
        print(f"[DEBUG] ClinicalNER sample: {df_clin[sample_cols].head(3).to_dict('records')}")
    
    df_med7 = med7_model.process_dataframe(df_sentences)
    print(f"[DEBUG] After Med7NER: {len(df_med7)} entities")
    
    df_entities = pd.concat([df_clin, df_med7], ignore_index=True)
    print(f"[DEBUG] After concat: {len(df_entities)} total entities")

    if df_entities.empty:
        print("[DEBUG] No entities found - returning empty result")
        return {"entities": [], "relations": [], "graph": {}, "edges": []}

    merged_entities = merge_and_deduplicate(df_entities)
    print(f"[DEBUG] After merge_and_deduplicate: {len(merged_entities)} entities")
    if not merged_entities.empty:
        print(f"[DEBUG] Merged columns: {merged_entities.columns.tolist()}")
        sample_cols = ['word', 'entity_group'] if 'entity_group' in merged_entities.columns else merged_entities.columns.tolist()[:3]
        print(f"[DEBUG] Merged sample: {merged_entities[sample_cols].head(3).to_dict('records')}")
        # Removes wordpiece artifacts and short token fragments from API output.
        merged_entities = final_polish(merged_entities)
        print(f"[DEBUG] After final_polish: {len(merged_entities)} entities")
        if not merged_entities.empty:
            print(f"[DEBUG] Polished sample: {merged_entities[sample_cols].head(3).to_dict('records')}")

    # --- STAGE 3: Semantic Extraction ---
    # Note: Radiology reports are purely diagnostic (findings + impression) with no medications,
    # so drug-disease relation extraction is not applicable. Return entities only.
    df_normalized_dict = []
    graph_payload = {}
    edges_dict = []

    if mapped_type != "radiology":
        # Only extract drug-disease relations for discharge summaries
        # Primary threshold is 0.50
        df_relations = rel_extractor.extract_drug_disease(
            merged_entities, 
            df_sentences, 
            window=2, 
            valid_sections=ACTIVE_SECTIONS, 
            threshold=0.50
        )

        if df_relations.empty:
            print("No relations found with strict filters; retrying with relaxed window/sections...")
            df_relations = rel_extractor.extract_drug_disease(
                merged_entities,
                df_sentences,
                window=4,
                valid_sections=None,
                threshold=0.50,
            )

        # --- STAGES 4 & 5: UMLS Mapping & Graph Building ---
        if not df_relations.empty:
            # FIX 3: Lower secondary filter to 0.60 to capture more NLI results
            df_relations_filtered = df_relations[
                (df_relations['relation_type'].isin(['CAUSES', 'TREATS'])) & 
                (df_relations['model_confidence'] >= 0.60)
            ].copy()

            if not df_relations_filtered.empty:
                df_normalized = mapper.map_dataframe(df_relations_filtered)
                if 'section_name' not in df_normalized.columns:
                    df_normalized['section_name'] = 'unknown'
                df_normalized['target_semantic_valid'] = df_normalized.apply(
                    lambda row: _validate_target_semantic_type(linker, row['cui_2'], row['relation_type']),
                    axis=1,
                )
                df_normalized_dict = df_normalized.to_dict(orient="records")
                
                if not df_normalized.empty and 'cui_1' in df_normalized.columns:
                    df_joined = joiner.build_longitudinal_edges(df_normalized)
                    
                    if not df_joined.empty:
                        df_scored = scorer.score_edges(df_joined)
                        graph_payload = graph_builder.build_json_graph(df_scored)
                        edges_dict = df_scored.to_dict(orient="records")

    result = {
        "entities": merged_entities.to_dict(orient="records"),
        "relations": df_normalized_dict,
        "graph": graph_payload,
        "edges": edges_dict
    }
    
    # For radiology reports, add enhanced structured data
    if mapped_type == "radiology":
        rad_data = rad_enhancer.enhance_entities_dataframe(merged_entities, df_sentences, df_raw)
        finding_relations = rad_data.get("finding_relationships", [])
        severity_map = {
            str(k).strip().lower(): v
            for k, v in (rad_data.get("severity_classification", {}) or {}).items()
        }
        radiology_edges = _build_radiology_edges(
            relations=finding_relations,
            hadm_id=hadm_id,
            note_id=file.filename,
            subject_id=subject_id,
            severity_map=severity_map,
            confidence_threshold=0.50,  # Only include relations with >= 50% confidence
        )

        result["relations"] = finding_relations
        result["edges"] = radiology_edges
        result["graph"] = {
            str(hadm_id): {
                "nodes": list({
                    e.get("word", "").strip().lower()
                    for e in merged_entities.to_dict(orient="records")
                    if str(e.get("word", "")).strip()
                }),
                "edges": [
                    {
                        "source": edge["entity_1"],
                        "target": edge["entity_2"],
                        "relation_type": edge["relation_type"],
                        "confidence": edge["model_confidence"],
                    }
                    for edge in radiology_edges
                ],
            }
        }

        result["measurements"] = rad_data.get("measurements", [])
        result["negations"] = rad_data.get("negations", [])
        result["finding_relationships"] = finding_relations
        result["section_summary"] = rad_data.get("section_summary", {})
        result["severity_classification"] = rad_data.get("severity_classification", {})
    
    return _to_builtin_types(result)