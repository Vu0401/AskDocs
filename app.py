import os
import sys
import hashlib
import base64
import streamlit as st

# Import necessary modules
from rag.rag import RAG
from rag.vectordb import VectorDB
from util.util import extract_text_from_pdf, chunk_text, convert_to_documents

# Workaround to ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def get_image_as_base64(image_path):
    """Load and encode an image as base64."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def initialize_session_state():
    """Initialize session state variables for chat, PDF processing, and RAG system."""
    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""
    if 'pdf_chunks' not in st.session_state:
        st.session_state.pdf_chunks = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'rag' not in st.session_state:
        st.session_state.rag = RAG(model="gemini/gemini-2.0-flash-thinking-exp-01-21", functions=[])
    if 'relevant_docs' not in st.session_state:
        st.session_state.relevant_docs = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDB()
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}


def get_file_hash(file_content):
    """Calculate the MD5 hash of a file content to track duplicates."""
    return hashlib.md5(file_content).hexdigest()



def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="AskDocs", page_icon="askdocs.jpg", layout="wide")
    initialize_session_state()

    # Sidebar for file upload
    with st.sidebar:
        logo_path = os.path.join("assets", "askdocs.jpg")
        img_str = get_image_as_base64(logo_path)

        if img_str:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/jpeg;base64,{img_str}" alt="AskDocs Logo" width="300" 
                    style="border-radius: 20px; border: 6px solid #D80070; box-shadow: 0px 0px 15px rgba(216, 0, 112, 0.8);">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">AskDocs</div>""", 
                unsafe_allow_html=True
            )
        
        st.header("📂 Upload Your PDFs")
        uploaded_files = st.file_uploader("📤 Drag & Drop or Select PDF Files", type=['pdf'], accept_multiple_files=True)
        process_button = st.button("🛠 Process PDFs")

    # Processing uploaded PDFs
    if process_button and uploaded_files:
        with st.spinner('⚙️ Extracting & Analyzing PDFs...'):
            all_chunks = {}
            new_files_count, duplicate_files_count = 0, 0

            for pdf_file in uploaded_files:
                pdf_content = pdf_file.read()
                file_hash = get_file_hash(pdf_content)

                if file_hash in st.session_state.processed_files:
                    duplicate_files_count += 1
                    continue
                
                st.session_state.processed_files[file_hash] = pdf_file.name
                new_files_count += 1

                pdf_file.seek(0)
                text = extract_text_from_pdf(pdf_file)
                chunks = chunk_text(text)
                
                for chunk in chunks:
                    chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                    if chunk_id not in all_chunks:
                        all_chunks[chunk_id] = chunk

            if new_files_count > 0 and all_chunks:
                doc_chunks = convert_to_documents(list(all_chunks.values()))
                st.session_state.vector_db.add_documents(doc_chunks)
                st.session_state.retriever = st.session_state.vector_db.get_retriever()

        with st.sidebar:
            if new_files_count > 0:
                st.success(f"✅ Successfully processed {new_files_count} new PDF(s)!")
            if duplicate_files_count > 0:
                st.warning(f"ℹ️ Skipped {duplicate_files_count} duplicate file(s)")

    if not st.session_state.retriever:
        st.warning("📌 Please upload and process PDFs before asking questions.")
        return

    col_main, col_relevant = st.columns([2, 1])

    with col_main:
        question = st.chat_input("💭 Type your question here...", key="chat_input_top")
        chat_container = st.container(height=500)

        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
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
            recent_messages = st.session_state.chat_history[-5:]
            answer = st.session_state.rag.llm(messages=recent_messages, context=context)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(answer)

    with col_relevant:
        st.markdown("""<h3 style='color: #E0E0E0;'>🔍 Relevant Documents</h3>""", unsafe_allow_html=True)

        if not st.session_state.relevant_docs:
            st.markdown("""<div style='background-color: #2A2A2A; padding: 15px; border-radius: 8px; text-align: center; color: #A0A0A0;'>📄 No relevant documents found yet.</div>""", unsafe_allow_html=True)
        else:
            with st.container(height=500):
                for doc in st.session_state.relevant_docs:
                    with st.expander(f"{doc.page_content[:30]}", expanded=False):
                        st.write(doc.page_content)

if __name__ == "__main__":
    main()
