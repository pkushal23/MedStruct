# MedStruct

MedStruct is a clinical NLP pipeline that converts unstructured medical PDFs into structured entities, normalized relationships, and admission-level knowledge graphs. The project is built around discharge summaries and radiology reports, with separate downstream logic for each document type.

## What It Does

- Extracts text from uploaded PDF notes
- Cleans and segments clinical text into sections and sentences
- Runs an ensemble NER pipeline using clinical and medication-focused models
- Extracts semantic relationships from discharge summaries
- Maps high-confidence relations to UMLS concepts
- Builds cross-note longitudinal graph edges and JSON knowledge graphs
- Adds radiology-specific outputs such as finding relationships, negations, measurements, and severity labels

## Pipeline Overview

### Stage 1: Text Processing

- Loads notes from MIMIC-IV or accepts uploaded PDFs
- Normalizes OCR and PDF artifacts
- Segments text into clinical sections
- Tokenizes sections into sentence-level records
- Harmonizes ICD diagnosis codes for batch workflows

Artifacts:
- `data/processed/sections.csv`
- `data/processed/sentences.csv`
- `data/processed/diagnoses_icd_mapped.csv`

### Stage 2: Entity Extraction

- Uses `ClinicalNER` for general clinical entities
- Uses `Med7NER` for medication-focused extraction
- Merges overlapping detections and removes duplicates
- Applies final polishing to reduce token fragment artifacts

Artifacts:
- `data/processed/entities.csv`
- `data/processed/entities_refined.csv`

### Stage 3: Relation Extraction

- Focuses on drug-disease relations in discharge summaries
- Filters by targeted clinical sections and model confidence
- Produces verified semantic relations such as `CAUSES` and `TREATS`

Artifacts:
- `data/processed/relations_verified.csv`

### Stage 4: Ontology Normalization

- Maps high-confidence relations to UMLS concepts
- Uses SciSpacy-based ontology linking
- Produces normalized concept-level relations

Artifacts:
- `data/processed/relations_normalized.csv`

### Stage 5: Longitudinal Graph Building

- Aligns normalized relations across notes
- Scores cross-note alignment strength
- Builds admission-level graph JSON and edge tables

Artifacts:
- `data/processed/longitudinal_edges.csv`
- `data/processed/longitudinal_graphs.json`

## Supported Document Types

### Discharge Summaries

- Full 5-stage processing
- Entity extraction
- Drug-disease relation extraction
- UMLS normalization
- Knowledge graph generation

### Radiology Reports

- Stage 1 and Stage 2 extraction
- Radiology-specific analysis instead of drug-disease relation extraction
- Structured findings, measurements, negations, severity classification, and finding-to-finding edges

## Project Structure

```text
MedStruct/
|-- main.py                # FastAPI backend for PDF processing
|-- ui.py                  # Streamlit frontend
|-- stage_1/               # Cleaning, segmentation, tokenization, ICD mapping
|-- stage_2/               # NER models, merging, output polishing, radiology enrichment
|-- stage_3/               # Relation extraction
|-- stage_4/               # UMLS mapping and ontology validation
|-- stage_5/               # Cross-note alignment and graph building
|-- src/                   # Earlier pipeline utilities and BigQuery helpers
|-- data/
|   |-- external/          # Reference mapping files
|   `-- processed/         # Generated CSV and JSON outputs
`-- requirements.txt
```

## Requirements

- Python 3.10 or newer recommended
- Windows, macOS, or Linux
- Access to the required NLP model weights listed in `requirements.txt`
- Optional: Google Cloud credentials and BigQuery access for batch note loading from MIMIC-IV

## Environment Variables

Create a `.env` file with the values your environment needs.

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
HUGGINGFACEHUB_API_TOKEN=your-token-if-required
```

Notes:
- `GOOGLE_CLOUD_PROJECT` is used by the batch loaders under `stage_1/` and `src/`
- UMLS normalization also requires the SciSpacy medical model used by Stage 4

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you are using PowerShell on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the App

Start the FastAPI backend:

```bash
uvicorn main:app --reload
```

In a second terminal, start the Streamlit UI:

```bash
streamlit run ui.py
```

Then:

1. Open the Streamlit app in your browser
2. Enter `subject_id` and `hadm_id`
3. Choose either `Discharge Summary` or `Radiology Report`
4. Upload a PDF
5. Run the pipeline and review the extracted tables and graph output

The UI saves exported CSV files under:

```text
output/ui_exports/<pdf-name>/
```

## API Usage

`POST /process-pdf`

Form fields:
- `file`: PDF file
- `subject_id`: integer
- `hadm_id`: integer
- `note_type`: `discharge` or `radiology`

Response includes:
- `entities`
- `relations`
- `edges`
- `graph`

Radiology responses may also include:
- `measurements`
- `negations`
- `finding_relationships`
- `section_summary`
- `severity_classification`

## Running Stages Individually

You can run each batch stage directly:

```bash
python stage_1/run_stage-1.py
python stage_2/run_stage-2.py
python stage_3/run_stage-3.py
python stage_4/run_stage-4.py
python stage_5/run_stage-5.py
```

## Data Sources

- MIMIC-IV note tables for discharge summaries and radiology reports
- MIMIC-IV diagnoses for ICD harmonization
- UMLS and SciSpacy resources for ontology linking

## Current Notes and Limitations

- The current radiology path does not use a dedicated RadBERT NER classifier because the checkpoint lacks trained classifier weights in this project setup
- Relation extraction is focused on discharge-summary medication and problem relationships
- Some pipeline modules assume local model availability and a configured NLP environment
- Batch loaders depend on valid Google Cloud authentication and dataset access

## Outputs at a Glance

- `entities`: merged clinical entities
- `relations`: normalized discharge relations or radiology finding relationships
- `edges`: graph edges with confidence or alignment scores
- `graph`: admission-level JSON graph for downstream use

## License

Add the project license here if you plan to distribute or publish the repository.
