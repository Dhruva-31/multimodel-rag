import tempfile

import streamlit as st

from pipelines.ingestion_pipeline import (
    build_knowledge_base,
)

from pipelines.query_pipeline import (
    ask_question,
)

st.set_page_config(
    page_title="Multimodal Knowledge Engine",
    layout="wide",
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "ready": False,
    "indexing": False,
    "uploaded_file": None,
    "bm25": None,
    "chroma": None,
    "uploader_key": 0,
    "messages": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# TITLE
# =====================================================

st.title("Multimodal Knowledge Engine")

st.caption("Upload a resource and query it using Hybrid RAG")


# =====================================================
# LAYOUT
# =====================================================

left, right = st.columns([1, 3])


# =====================================================
# LEFT PANEL
# =====================================================

with left:

    st.subheader("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload Resource",
        key=f"uploader_{st.session_state.uploader_key}",
        type=[
            "pdf",
            "txt",
            "docx",
            "png",
            "jpg",
            "jpeg",
        ],
        disabled=(st.session_state.ready or st.session_state.indexing),
    )

    # ==========================================
    # BUILD KNOWLEDGE BASE
    # ==========================================

    if uploaded_file and not st.session_state.ready and not st.session_state.indexing:

        st.session_state.indexing = True

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="_" + uploaded_file.name,
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())

            file_path = tmp.name

        status = st.status(
            "Building knowledge base...",
            expanded=True,
        )

        try:

            status.write("Loading resource...")

            status.write("Chunking...")

            status.write("Generating embeddings...")

            status.write("Building vector index...")

            status.write("Building BM25 index...")

            status.write("Initializing retriever...")

            bm25, chroma = build_knowledge_base(
                file_path,
                original_filename=uploaded_file.name,
            )

            st.session_state.bm25 = bm25
            st.session_state.chroma = chroma

            st.session_state.ready = True

            st.session_state.uploaded_file = uploaded_file.name

            status.update(
                label="Knowledge base ready",
                state="complete",
            )

        except Exception as e:

            status.update(
                label=str(e),
                state="error",
            )

        finally:

            st.session_state.indexing = False

    # ==========================================
    # STATUS
    # ==========================================

    st.divider()

    if st.session_state.ready:

        st.success(f"Loaded:\n" f"{st.session_state.uploaded_file}")

    else:

        st.info("No resource loaded")

    # ==========================================
    # CLEAR
    # ==========================================

    if st.button(
        "Clear Session",
        use_container_width=True,
    ):

        st.session_state.ready = False

        st.session_state.indexing = False

        st.session_state.messages = []

        st.session_state.uploaded_file = None

        st.session_state.bm25 = None

        st.session_state.chroma = None

        # resets file uploader
        st.session_state.uploader_key += 1

        if st.session_state.chroma:
            st.session_state.chroma.delete_collection()

        st.rerun()


# =====================================================
# RIGHT PANEL
# =====================================================


with right:

    st.subheader("Chat")

    if st.session_state.ready:

        st.caption(f"📄 {st.session_state.uploaded_file}")

    if not st.session_state.messages:

        st.info("Upload a resource and start asking questions.")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# =====================================================
# QUERY
# =====================================================


query = st.chat_input(
    "Ask a question...",
    disabled=not st.session_state.ready,
)

if query:

    # user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.spinner("Retrieving..."):

        response = ask_question(
            query,
            st.session_state.bm25,
        )

    # assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    st.rerun()
