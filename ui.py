import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile, os, time

st.set_page_config(layout="wide")
st.title("ClinExtract: Medical data structuring tool")

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
if "csv_bytes" not in st.session_state:
    st.session_state.csv_bytes = {}
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_submitted_at" not in st.session_state:
    st.session_state.job_submitted_at = None

# ── Phase 1: Submit job ────────────────────────────────────────────────────
if st.sidebar.button("Process Document") and uploaded_file:
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
    st.session_state.processed_result = None
    st.session_state.csv_bytes = {}
    st.session_state.job_id = None

    try:
        resp = requests.post(
            "http://localhost:8000/process-pdf",
            files=files, data=data, timeout=30
        )
        if resp.status_code == 200:
            st.session_state.job_id = resp.json()["job_id"]
            st.session_state.job_submitted_at = time.time()
        else:
            st.session_state.last_error = f"Submit failed ({resp.status_code}): {resp.text}"
    except Exception as exc:
        st.session_state.last_error = f"Connection failed: {exc}"

# ── Phase 2: Poll until done ───────────────────────────────────────────────
if st.session_state.job_id and st.session_state.processed_result is None:
    elapsed = int(time.time() - (st.session_state.job_submitted_at or time.time()))
    status_box = st.info(f"⏳ Processing… {elapsed}s elapsed. Page auto-updates every 3 s.")

    try:
        poll = requests.get(
            f"http://localhost:8000/job/{st.session_state.job_id}",
            timeout=10
        )
        job = poll.json()
    except Exception as exc:
        st.error(f"Polling failed: {exc}")
        job = {"status": "error", "error": str(exc)}

    if job["status"] == "done":
        result = job["result"]
        st.session_state.processed_result = result
        st.session_state.last_status_code = 200
        st.session_state.job_id = None

        output_dir = Path("output") / "ui_exports" / Path(uploaded_file.name if uploaded_file else "doc").stem
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        csv_bytes = {}
        for key in ["entities", "relations", "edges"]:
            df = pd.DataFrame(result.get(key, []))
            path = output_dir / f"{key}_{int(hadm_id)}.csv"
            df.to_csv(path, index=False)
            saved_paths[key] = str(path.resolve())
            csv_bytes[key] = df.to_csv(index=False).encode("utf-8")

        st.session_state.saved_csv_paths = saved_paths
        st.session_state.csv_bytes = csv_bytes
        st.success("✅ Processing complete!")
        st.rerun()

    elif job["status"] == "error":
        st.session_state.last_error = job.get("error", "Unknown error")
        st.session_state.job_id = None
        st.error(f"❌ Pipeline failed: {st.session_state.last_error}")

    else:  # pending or running — keep polling
        time.sleep(3)
        st.rerun()

# ── Sidebar download buttons (visible after any successful run) ────────────
if st.session_state.csv_bytes:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⬇ Download Results")
    hadm_val = int(st.session_state.last_request_payload.get("hadm_id", 0)) if st.session_state.last_request_payload else 0
    label_map = {
        "entities":  ("Entities CSV",  f"entities_{hadm_val}.csv"),
        "relations": ("Relations CSV", f"relations_{hadm_val}.csv"),
        "edges":     ("Edges CSV",     f"edges_{hadm_val}.csv"),
    }
    for key, (label, fname) in label_map.items():
        if key in st.session_state.csv_bytes:
            st.sidebar.download_button(
                label=f"📥 {label}",
                data=st.session_state.csv_bytes[key],
                file_name=fname,
                mime="text/csv",
                key=f"sidebar_dl_{key}",
            )

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
                df_edges = pd.DataFrame(result["edges"])

                # ── Metrics row ──────────────────────────────────────────────
                graph_data = result.get("graph", {})
                # graph_data is keyed by hadm_id; grab first admission
                admission_data = next(iter(graph_data.values()), {}) if graph_data else {}
                summary = admission_data.get("summary", {})
                nodes_list = admission_data.get("nodes", [])
                edges_list = admission_data.get("edges", [])

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Nodes", summary.get("total_nodes", len(nodes_list)))
                mcol2.metric("Edges", summary.get("total_edges", len(edges_list)))
                consensus_ratio = summary.get("consensus_ratio", 0)
                mcol3.metric("Consensus Ratio", f"{consensus_ratio:.0%}")
                mcol4.metric("Total Raw Edges", len(df_edges))

                st.markdown("---")

                # ── Interactive graph ────────────────────────────────────────
                if nodes_list and edges_list:
                    st.subheader("Knowledge Graph")
                    st.caption(
                        "🔵 Node size = centrality  |  "
                        "🟡 Amber edges = consensus (seen in multiple notes/sections)  |  "
                        "🩵 Teal edges = single-note  |  "
                        "Hover for details"
                    )

                    net = Network(
                        height="620px",
                        width="100%",
                        bgcolor="#0f1117",
                        font_color="#e0e0e0",
                        directed=True,
                    )
                    net.force_atlas_2based(
                        gravity=-50,
                        central_gravity=0.01,
                        spring_length=120,
                        spring_strength=0.08,
                        damping=0.4,
                        overlap=0,
                    )

                    # Add nodes — size scaled by centrality
                    max_centrality = max((n.get("centrality", 1) for n in nodes_list), default=1)
                    for node in nodes_list:
                        c = node.get("centrality", 1)
                        size = 14 + (c / max_centrality) * 28
                        net.add_node(
                            node["id"],
                            label=node["name"],
                            title=f"<b>{node['name']}</b><br>CUI: {node['id']}<br>Centrality: {c}",
                            size=size,
                            color={
                                "background": "#4f8ef7",
                                "border": "#1a56db",
                                "highlight": {"background": "#7eb3ff", "border": "#1a56db"},
                            },
                            font={"size": 12, "color": "#e0e0e0"},
                        )

                    # Add edges
                    for edge in edges_list:
                        is_consensus = edge.get("is_consensus", False)
                        score = edge.get("alignment_score", 0)
                        color = "#f59e0b" if is_consensus else "#2dd4bf"
                        sections = ", ".join(edge.get("evidence_sections", []))
                        tooltip = (
                            f"<b>{edge['relation']}</b><br>"
                            f"Score: {score:.3f}<br>"
                            f"Consensus: {'Yes' if is_consensus else 'No'}<br>"
                            f"Sections: {sections or 'N/A'}"
                        )
                        net.add_edge(
                            edge["source"],
                            edge["target"],
                            label=edge["relation"],
                            title=tooltip,
                            color=color,
                            width=1.5 + score * 2,
                            arrows="to",
                            font={"size": 9, "color": "#b0b0b0", "strokeWidth": 0},
                        )

                    # Render to temp HTML and embed
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
                        net.save_graph(tmp.name)
                        html_content = open(tmp.name, "r", encoding="utf-8").read()
                    os.unlink(tmp.name)

                    components.html(html_content, height=640, scrolling=False)
                else:
                    st.info("Graph node/edge data not available in API response. Showing raw edges below.")

                st.markdown("---")

                # ── Inline CSV downloads ─────────────────────────────────────
                if st.session_state.csv_bytes:
                    dcol1, dcol2, dcol3 = st.columns(3)
                    hadm_val = int(st.session_state.last_request_payload.get("hadm_id", 0)) if st.session_state.last_request_payload else 0
                    with dcol1:
                        if "entities" in st.session_state.csv_bytes:
                            st.download_button(
                                "📥 Download Entities CSV",
                                data=st.session_state.csv_bytes["entities"],
                                file_name=f"entities_{hadm_val}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    with dcol2:
                        if "relations" in st.session_state.csv_bytes:
                            st.download_button(
                                "📥 Download Relations CSV",
                                data=st.session_state.csv_bytes["relations"],
                                file_name=f"relations_{hadm_val}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    with dcol3:
                        if "edges" in st.session_state.csv_bytes:
                            st.download_button(
                                "📥 Download Edges CSV",
                                data=st.session_state.csv_bytes["edges"],
                                file_name=f"edges_{hadm_val}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                with st.expander("📋 Raw Edge Table", expanded=False):
                    st.dataframe(df_edges, use_container_width=True)
                with st.expander("🗂 Graph JSON", expanded=False):
                    st.json(graph_data)
            else:
                st.write("No edges formed.")