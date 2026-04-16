from fastapi import FastAPI, UploadFile, File, Form
import fitz
import pandas as pd
import sys



from stage_1.text_cleaner import clean_dataframe
from stage_1.section_segmenter import segment_dataframe
from stage_1.tokenizer import ClinicalTokenizer
from stage_2.clinical_ner import ClinicalNER
from stage_2.med7_ner import Med7NER
from stage_2.radbert_ner import RadBERTNER
from stage_2.entity_merger import merge_and_deduplicate, stitch_subwords
from stage_3.relation_extractor import RelationExtractor
from stage_4.cui_mapper import CUIMapper
from stage_4.ontology_validation import OntologyLinker
from stage_5.cross_note_alignment import CrossNoteJoiner
from stage_5.alignment_scorer import AlignmentScorer
from stage_5.entity_graph_builder import EntityGraphBuilder

app = FastAPI()

tokenizer = ClinicalTokenizer()
clinical_model = ClinicalNER()
med7_model = Med7NER()
radbert_model = RadBERTNER()
rel_extractor = RelationExtractor(drug_labels=['DRUG', 'treatment'], disease_labels=['problem'], device=-1)
linker = OntologyLinker()
mapper = CUIMapper(linker)
joiner = CrossNoteJoiner()
scorer = AlignmentScorer(multi_note_boost=0.05)
graph_builder = EntityGraphBuilder()

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

    df_raw = pd.DataFrame([{
        "subject_id": subject_id, 
        "hadm_id": hadm_id, 
        "note_id": file.filename, 
        "text": text
    }])

    # STAGE 1
    df_clean = clean_dataframe(df_raw)
    df_seg = segment_dataframe(df_clean, note_type)
    df_sections, df_sentences = tokenizer.tokenize_dataframe(df_seg)

    # STAGE 2
    if note_type == "radiology":
        df_entities = radbert_model.process_dataframe(df_sentences)
        if not df_entities.empty and 'entity_group' in df_entities.columns:
            df_entities = df_entities[~df_entities['entity_group'].str.contains('LABEL')]
    else:
        df_clin = clinical_model.process_dataframe(df_sentences)
        df_med7 = med7_model.process_dataframe(df_sentences)
        df_entities = pd.concat([df_clin, df_med7], ignore_index=True)

    if df_entities.empty:
        return {"entities": [], "relations": [], "graph": {}}

    stitched_entities = stitch_subwords(df_entities)
    merged_entities = merge_and_deduplicate([stitched_entities])

    # STAGE 3
    df_relations = rel_extractor.extract_drug_disease(merged_entities, df_sentences, window=1, threshold=0.7)

    # STAGES 4 & 5 (Only run if relations were found)
    df_normalized_dict = []
    graph_payload = {}
    edges_dict = []

    if not df_relations.empty:
        df_normalized = mapper.map_dataframe(df_relations)
        df_normalized_dict = df_normalized.to_dict(orient="records")
        
        # Double check mapping didn't result in an empty set before proceeding to Stage 5
        if not df_normalized.empty and 'cui_1' in df_normalized.columns:
            df_joined = joiner.build_longitudinal_edges(df_normalized)
            if not df_joined.empty:
                df_scored = scorer.score_edges(df_joined)
                graph_payload = graph_builder.build_json_graph(df_scored)
                edges_dict = df_scored.to_dict(orient="records")

    return {
        "entities": merged_entities.to_dict(orient="records"),
        "relations": df_normalized_dict,
        "graph": graph_payload,
        "edges": edges_dict
    }