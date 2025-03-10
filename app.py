import streamlit as st
import hashlib
import base64
import os
from pathlib import Path
from PIL import Image

from rag.rag import RAG
from rag.vectordb import VectorDB
from util.util import extract_text_from_pdf, chunk_text, convert_to_documents

# Workaround to ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def get_image_as_base64(image_path):
    """Load and encode an image as base64 for displaying."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def initialize_session_state():
    """Initialize session state variables for chat, PDF processing, and RAG system."""
    defaults = {
        "chat_input": "",
        "pdf_chunks": [],
        "chat_history": [],
        "retriever": None,
        "rag": RAG(model="gemini/gemini-2.0-flash-thinking-exp-01-21", functions=[]),
        "relevant_docs": [],
        "vector_db": VectorDB(),
        "processed_files": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_file_hash(file_content):
    """Calculate the MD5 hash of file content."""
    return hashlib.md5(file_content).hexdigest()


def process_uploaded_files(uploaded_files):
    """Extract text from PDFs, chunk text, and store unique chunks in the vector database."""
    all_chunks = {}
    new_files, duplicates = 0, 0

    for pdf_file in uploaded_files:
        pdf_content = pdf_file.read()
        file_hash = get_file_hash(pdf_content)

        if file_hash in st.session_state.processed_files:
            duplicates += 1
            continue

        st.session_state.processed_files[file_hash] = pdf_file.name
        new_files += 1
        pdf_file.seek(0)
        chunks = chunk_text(extract_text_from_pdf(pdf_file))

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk

    if new_files and all_chunks:
        doc_chunks = convert_to_documents(list(all_chunks.values()))
        st.session_state.vector_db.add_documents(doc_chunks)
        st.session_state.retriever = st.session_state.vector_db.get_retriever()

    return new_files, duplicates


def display_processing_status(new_files, duplicates):
    """Display messages about file processing status in the sidebar."""
    if new_files:
        st.sidebar.success(f"✅ Successfully processed {new_files} new PDF(s)!")
    if duplicates:
        st.sidebar.warning(f"ℹ️ Skipped {duplicates} duplicate file(s)")


def main():
    st.set_page_config(page_title="AskDocs", page_icon=Image.open("assets/askdocs.jpg"), layout="wide")
    initialize_session_state()

    with st.sidebar:
        logo_path = "assets/askdocs.jpg"
        img_str = get_image_as_base64(logo_path)
        if img_str:
            st.markdown(f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{img_str}" '
                        'alt="AskDocs Logo" width="300" style="border-radius: 20px; border: 6px solid #D80070; '
                        'box-shadow: 0px 0px 15px rgba(216, 0, 112, 0.8);"></div>', unsafe_allow_html=True)
        else:
            st.sidebar.title("AskDocs")

        st.header("📂 Upload Your PDFs")
        uploaded_files = st.file_uploader("📤 Drag & Drop or Select PDF Files", type=['pdf'], accept_multiple_files=True)
        if st.button("🛠 Process PDFs") and uploaded_files:
            with st.spinner('⚙️ Extracting & Analyzing PDFs...'):
                new_files, duplicates = process_uploaded_files(uploaded_files)
            display_processing_status(new_files, duplicates)

    if not st.session_state.retriever:
        st.warning("📌 Please upload and process PDFs before asking questions.")
        return

    col_main, col_relevant = st.columns([2, 1])

    with col_main:
        question = st.chat_input("💭 Type your question here...", key="chat_input_top")
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                st.markdown("<br>", unsafe_allow_html=True)

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with chat_container:
                with st.chat_message("user"):
                    st.write(question)

            relevant_docs = st.session_state.retriever.invoke(question)
            seen, unique_docs = set(), []
            for doc in relevant_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            st.session_state.relevant_docs = unique_docs
            context = "\n".join([doc.page_content for doc in unique_docs])
            answer = st.session_state.rag.llm(messages=st.session_state.chat_history[-5:], context=context)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(answer)

    with col_relevant:
        st.markdown("<h3 style='color: #E0E0E0;'>🔍 Relevant Documents</h3>", unsafe_allow_html=True)
        if not st.session_state.relevant_docs:
            st.markdown("<div style='background-color: #2A2A2A; padding: 15px; border-radius: 8px; text-align: center; color: #A0A0A0;'>"
                        "📄 No relevant documents found yet.<br><small style='color: #808080;'>Try asking a question to retrieve related content!</small></div>",
                        unsafe_allow_html=True)
        else:
            with st.container(height=500):
                for doc in st.session_state.relevant_docs:
                    with st.expander(f"{doc.page_content[:30]}", expanded=False):
                        st.markdown(f"<div style='background-color: #2A2A2A; padding: 10px; border-radius: 8px;'>"
                                    f"<p style='color: #B0B0B0; font-size: 14px;'>{doc.page_content}</p></div>",
                                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()

