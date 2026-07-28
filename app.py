import streamlit as st
import tempfile
import os
from ingestion import (
    load_embedding_model, get_chroma_collection,
    ingest_pdf, get_ingested_sources, delete_source
)
from retrieval import retrieve_chunks
from generation import generate_answer

st.set_page_config(
    page_title="Equity Research Intelligence",
    page_icon="📊",
    layout="wide"
)

if "model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.model = load_embedding_model()

if "collection" not in st.session_state:
    st.session_state.collection = get_chroma_collection()

st.title("📊 Equity Research Report Intelligence")
st.caption("Upload equity research PDFs and ask questions across reports")

with st.sidebar:

    st.header("Upload Reports")
    uploaded_files = st.file_uploader(
        "Upload PDF reports",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        already_ingested = get_ingested_sources(st.session_state.collection)

        for uploaded_file in uploaded_files:
            if uploaded_file.name in already_ingested:
                st.info(f"✓ {uploaded_file.name} already ingested")
            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    count, message = ingest_pdf(
                        tmp_path,
                        st.session_state.model,
                        st.session_state.collection,
                        source_name=uploaded_file.name
                    )
                    os.unlink(tmp_path)
                    st.success(message)

    st.divider()
    st.header("Loaded Reports")

    loaded_sources = get_ingested_sources(st.session_state.collection)

    if not loaded_sources:
        st.caption("No reports loaded yet.")
    else:
        st.caption(f"{len(loaded_sources)} report(s) in database")
        for source in sorted(loaded_sources):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 {source}")
            if col2.button("🗑", key=f"del_{source}", help=f"Remove {source}"):
                removed = delete_source(st.session_state.collection, source)
                st.success(f"Removed {removed} chunks from {source}")
                st.rerun()

    st.divider()
    st.header("Filter by Report")
    st.caption(
        "Leave blank to search all reports. "
        "Select one to restrict the answer to that report only."
    )

    filter_options = ["All reports"] + sorted(loaded_sources)
    selected_filter = st.selectbox(
        "Search in:",
        options=filter_options,
        index=0
    )
    source_filter = (
        None if selected_filter == "All reports" else selected_filter
    )

st.header("Ask a Question")

query = st.text_input(
    "Ask anything about the uploaded reports",
    placeholder="What is the target price for Infosys?"
)

if query:
    with st.spinner("Searching reports..."):
        chunks = retrieve_chunks(
            query,
            st.session_state.model,
            st.session_state.collection,
            source_filter=source_filter
        )

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, chunks)

    st.markdown("### Answer")
    st.write(answer)

    if chunks:
        with st.expander("View retrieved sources"):
            for i, chunk in enumerate(chunks):
                meta = chunk["metadata"]
                st.markdown(
                    f"**Source {i+1}** — "
                    f"{meta['source']}, "
                    f"Page {meta['page']}, "
                    f"Type: {meta['type']} "
                    f"(distance: {chunk['distance']:.3f})"
                )
                st.caption(chunk["text"][:300] + "...")