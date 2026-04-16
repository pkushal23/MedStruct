import streamlit as st
import requests
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("MedStruct Advanced Pipeline Prototype")

NOTE_TYPE_OPTIONS = {
    "Discharge Summary": "DS",
    "Radiology Report": "RR",
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
        st.session_state.last_request_payload = {
            "file_name": uploaded_file.name,
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "note_type": note_type
        }

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

                df_entities = pd.DataFrame(result.get("entities", []))
                entities_path = output_dir / f"entities_{int(hadm_id)}.csv"
                df_entities.to_csv(entities_path, index=False)
                saved_paths["entities"] = str(entities_path.resolve())

                df_relations = pd.DataFrame(result.get("relations", []))
                relations_path = output_dir / f"relations_{int(hadm_id)}.csv"
                df_relations.to_csv(relations_path, index=False)
                saved_paths["relations"] = str(relations_path.resolve())

                df_edges = pd.DataFrame(result.get("edges", []))
                edges_path = output_dir / f"edges_{int(hadm_id)}.csv"
                df_edges.to_csv(edges_path, index=False)
                saved_paths["edges"] = str(edges_path.resolve())

                st.session_state.saved_csv_paths = saved_paths
                st.success("Processing complete.")
            else:
                st.error(f"Failed to process the document. Status code: {response.status_code}")
                st.session_state.last_error = response.text
                st.session_state.processed_result = None
        except Exception as exc:
            st.error("Failed to process the document.")
            st.session_state.last_error = str(exc)
            st.session_state.processed_result = None

if st.session_state.processed_result:
    result = st.session_state.processed_result
    relation_count = len(result.get("relations", []))
    edge_count = len(result.get("edges", []))
    entity_count = len(result.get("entities", []))

    st.write(f"Status code: {st.session_state.last_status_code}")
    st.write(f"Entities: {entity_count} | Relations: {relation_count} | Edges: {edge_count}")

    if st.session_state.last_request_payload:
        with st.expander("Request Payload", expanded=True):
            st.json(st.session_state.last_request_payload)
            if st.session_state.last_request_payload.get("note_type") == "RR":
                st.warning("This request was sent as `RR`, which uses the radiology branch in the backend and can produce far fewer entities and no relations for this PDF.")

    if st.session_state.saved_csv_paths:
        with st.expander("Saved CSV Files", expanded=True):
            for label, path in st.session_state.saved_csv_paths.items():
                st.code(f"{label}: {path}")
    
    tab1, tab2, tab3 = st.tabs(["Stage 2: Entities", "Stage 3 & 4: Normalized Relations", "Stage 5: Knowledge Graph"])
    
    with tab1:
        if result.get("entities"):
            df_entities = pd.DataFrame(result["entities"])
            st.dataframe(df_entities)
            
            # Convert to CSV and create download button
            csv_entities = df_entities.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Entities as CSV",
                data=csv_entities,
                file_name=f"medstruct_entities_{hadm_id}.csv",
                mime="text/csv",
            )
        else:
            st.write("No entities extracted.")
            
    with tab2:
        if result.get("relations"):
            df_relations = pd.DataFrame(result["relations"])
            st.dataframe(df_relations)
            
            # Convert to CSV and create download button
            csv_relations = df_relations.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Relations as CSV",
                data=csv_relations,
                file_name=f"medstruct_relations_{hadm_id}.csv",
                mime="text/csv",
            )
        else:
            st.write("No relations extracted.")
            
    with tab3:
        if result.get("edges"):
            df_edges = pd.DataFrame(result["edges"])
            st.dataframe(df_edges)
            
            # Convert to CSV and create download button
            csv_edges = df_edges.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Longitudinal Edges as CSV",
                data=csv_edges,
                file_name=f"medstruct_longitudinal_edges_{hadm_id}.csv",
                mime="text/csv",
            )
        else:
            st.write("No edges formed.")
            
        with st.expander("View Raw Graph JSON"):
            st.json(result.get("graph", {}))

    with st.expander("View Raw API Response"):
        st.json(result)
elif st.session_state.last_error:
    st.write(f"Status code: {st.session_state.last_status_code}")
    with st.expander("Request Error Details", expanded=True):
        st.text(st.session_state.last_error)
