import streamlit as st
import requests
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("MedStruct Advanced Pipeline Prototype")

NOTE_TYPE_OPTIONS = {
    "Discharge Summary": "discharge",
    "Radiology Report": "radiology",
}

st.sidebar.header("Upload Document")
subject_id = st.sidebar.number_input("Subject ID", min_value=1, step=1)
hadm_id = st.sidebar.number_input("Hospital Admission ID (HADM ID)", min_value=1, step=1)
note_type_label = st.sidebar.selectbox("Document Type", list(NOTE_TYPE_OPTIONS.keys()))
uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

if "processed_result" not in st.session_state:
    st.session_state.processed_result = None
if "last_status_code" not in st.session_state:
    st.session_state.last_status_code = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "saved_csv_paths" not in st.session_state:
    st.session_state.saved_csv_paths = {}
if "last_request_payload" not in st.session_state:
    st.session_state.last_request_payload = None

if st.sidebar.button("Process Document") and uploaded_file:
    with st.spinner("Running Stages 1 through 5..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        note_type = NOTE_TYPE_OPTIONS[note_type_label]
        data = {
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "note_type": note_type
        }
        st.session_state.last_request_payload = data

        st.session_state.last_error = None
        st.session_state.saved_csv_paths = {}

        try:
            response = requests.post("http://localhost:8000/process-pdf", files=files, data=data, timeout=300)
            st.session_state.last_status_code = response.status_code

            if response.status_code == 200:
                result = response.json()
                st.session_state.processed_result = result

                output_dir = Path("output") / "ui_exports" / Path(uploaded_file.name).stem
                output_dir.mkdir(parents=True, exist_ok=True)

                saved_paths = {}
                for key in ["entities", "relations", "edges"]:
                    df = pd.DataFrame(result.get(key, []))
                    path = output_dir / f"{key}_{int(hadm_id)}.csv"
                    df.to_csv(path, index=False)
                    saved_paths[key] = str(path.resolve())

                st.session_state.saved_csv_paths = saved_paths
                st.success("Processing complete.")
            else:
                st.error(f"Failed to process. Status: {response.status_code}")
                st.session_state.last_error = response.text
                st.session_state.processed_result = None
        except Exception as exc:
            st.error("Connection failed.")
            st.session_state.last_error = str(exc)
            st.session_state.processed_result = None

if st.session_state.processed_result:
    result = st.session_state.processed_result
    st.write(f"Status: {st.session_state.last_status_code} | Entities: {len(result.get('entities', []))} | Relations: {len(result.get('relations', []))} | Edges: {len(result.get('edges', []))}")
    
    # Determine document type from payload
    note_type = st.session_state.last_request_payload.get('note_type', 'discharge') if st.session_state.last_request_payload else 'discharge'
    
    if note_type == "radiology":
        # RADIOLOGY-SPECIFIC TABS
        tab1, tab2, tab3 = st.tabs(["Stage-2: Entities", "Stage-2: Entities by Section", "Analysis & Recommendations"])

        with tab1:
            if result.get("entities"):
                st.subheader("All Extracted Clinical Entities")
                df_entities = pd.DataFrame(result["entities"])
                st.dataframe(df_entities, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Entities", len(result["entities"]))
                with col2:
                    entity_types = df_entities['entity_group'].value_counts() if 'entity_group' in df_entities.columns else {}
                    st.metric("Entity Types", len(entity_types))

                st.subheader("Entity Type Breakdown")
                if 'entity_group' in df_entities.columns:
                    st.bar_chart(df_entities['entity_group'].value_counts())
            else:
                st.write("No entities found.")

        with tab2:
            if result.get("entities"):
                df_entities = pd.DataFrame(result["entities"])
                if 'sentence_index' in df_entities.columns:
                    st.subheader("Entities Grouped by Sentence Context")
                    for sent_idx in sorted(df_entities['sentence_index'].unique()):
                        sent_entities = df_entities[df_entities['sentence_index'] == sent_idx]
                        section = sent_entities['section_name'].iloc[0] if 'section_name' in sent_entities.columns else "Unknown"
                        st.write(f"**{section} (Sentence {sent_idx})**")
                        st.dataframe(sent_entities[['word', 'entity_group', 'score']], use_container_width=True)
                else:
                    st.write("Section information not available.")
            else:
                st.write("No entities found.")

        with tab3:
            st.subheader("Radiology Report Analysis")
            if result.get("entities"):
                st.write("""
                #### Key Information Extracted
                - **Findings**: Clinical entities identified from the report
                - **Entity Types**: Problems (diagnoses) and Tests (procedures/findings)
                - **Negations**: Look for "No evidence of..." patterns in Impression
                """)

                if any('note_id' in e for e in result.get("entities", [])):
                    st.write("#### Document Metadata")
                    doc_meta = result["entities"][0]
                    st.write(f"- **Note ID**: {doc_meta.get('note_id')}")
                    st.write(f"- **HADM ID**: {doc_meta.get('hadm_id')}")
                    st.write(f"- **Entities Extracted**: {len(result['entities'])}")

                if result.get("measurements"):
                    st.subheader("Extracted Measurements & Dimensions")
                    measurements_list = []
                    for m in result.get("measurements", []):
                        measurements_list.append({
                            "Measurement": m.get("measurement"),
                            "Unit": m.get("unit"),
                            "Dimensions": str(m.get("dimensions"))
                        })
                    if measurements_list:
                        st.dataframe(pd.DataFrame(measurements_list), use_container_width=True)

                if result.get("negations"):
                    st.subheader("Normal Findings (Negated Findings)")
                    negations_list = []
                    for n in result.get("negations", []):
                        negations_list.append({
                            "Type": n.get("negation_type"),
                            "Finding": n.get("finding"),
                            "Full Context": n.get("full_text")
                        })
                    if negations_list:
                        st.dataframe(pd.DataFrame(negations_list), use_container_width=True)

                if result.get("severity_classification"):
                    st.subheader("Finding Severity Classification")
                    severity_dict = result.get("severity_classification", {})
                    severity_counts = pd.Series(severity_dict).value_counts()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Abnormal Findings", severity_counts.get('abnormal', 0), delta=None)
                    with col2:
                        st.metric("Normal Findings", severity_counts.get('normal', 0), delta=None)
                    with col3:
                        st.metric("Uncertain Findings", severity_counts.get('uncertain', 0), delta=None)

                    if severity_counts.sum() > 0:
                        st.bar_chart(severity_counts)

                if result.get("finding_relationships"):
                    st.subheader("Clinical Correlations (Finding-to-Finding Relationships)")
                    rel_list = []
                    for rel in result.get("finding_relationships", []):
                        rel_list.append({
                            "Finding 1": rel.get("entity_1"),
                            "Type 1": rel.get("entity_type_1"),
                            "Relation": rel.get("relation_type"),
                            "Finding 2": rel.get("entity_2"),
                            "Type 2": rel.get("entity_type_2"),
                            "Confidence": f"{rel.get('confidence', 0):.2%}"
                        })
                    if rel_list:
                        st.dataframe(pd.DataFrame(rel_list), use_container_width=True)
                else:
                    st.write("No significant clinical correlations found.")
            else:
                st.write("No entities found for analysis.")

    else:
        # DISCHARGE SUMMARY TABS (original behavior)
        tab1, tab2, tab3 = st.tabs(["Stage-2: Entities", "Stage-3,4: Normalized Relations", "Stage-5: Knowledge Graph"])
        
        with tab1:
            if result.get("entities"):
                st.dataframe(pd.DataFrame(result["entities"]), use_container_width=True)
            else:
                st.write("No entities found.")
                
        with tab2:
            if result.get("relations"):
                st.dataframe(pd.DataFrame(result["relations"]), use_container_width=True)
            else:
                st.write("No relations found.")
                
        with tab3:
            if result.get("edges"):
                st.dataframe(pd.DataFrame(result["edges"]), use_container_width=True)
                with st.expander("Graph JSON"):
                    st.json(result.get("graph", {}))
            else:
                st.write("No edges formed.")