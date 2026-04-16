import streamlit as st
import requests
import pandas as pd

st.set_page_config(layout="wide")
st.title("MedStruct Advanced Pipeline Prototype")

st.sidebar.header("Upload Document")
subject_id = st.sidebar.number_input("Subject ID", min_value=1, step=1)
hadm_id = st.sidebar.number_input("Hospital Admission ID (HADM ID)", min_value=1, step=1)
note_type = st.sidebar.selectbox("Document Type", ["discharge", "radiology"])
uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

if "processed_result" not in st.session_state:
    st.session_state.processed_result = None

if st.sidebar.button("Process Document") and uploaded_file:
    with st.spinner("Running Stages 1 through 5..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        data = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "note_type": note_type
        }
        
        response = requests.post("http://localhost:8000/process-pdf", files=files, data=data)
        
        if response.status_code == 200:
            st.session_state.processed_result = response.json()
            st.success("Processing complete.")
        else:
            st.error("Failed to process the document.")
            st.session_state.processed_result = None

if st.session_state.processed_result:
    result = st.session_state.processed_result
    
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
            
        with st.expander("View Raw Graph JSON"):
            st.json(result.get("graph", {}))