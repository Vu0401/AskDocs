__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from rag.vectordb import VectorDB
from rag.rag import RAG
import hashlib
from util.util import extract_text_from_pdf, chunk_text, convert_to_documents

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print("Error: ", e)
        return None

# 🎯 Function to initialize session state
def initialize_session_state():
    """Initialize session state variables."""
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
        st.session_state.vector_db = None

# 🚀 Main function
def main():
    st.set_page_config(page_title="AskDocs", page_icon="🔍", layout="wide")

    # Initialize session state
    initialize_session_state()

    # 🎯 Sidebar for PDF upload (LEFT SIDEBAR)
    with st.sidebar:
        img_path = "assets/askdocs.jpg" 
        img_base64 = img_to_base64(img_path)
        if img_base64:
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 10px;">
                    <img src="data:image/png;base64,{img_base64}" width="100" style="border-radius: 10px; box-shadow: 0px 0px 10px rgba(80, 200, 120, 0.8);">
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        st.header("📂 Upload Your PDFs")
        uploaded_files = st.file_uploader(
            "📤 Drag & Drop or Select PDF Files",
            type=['pdf'],
            accept_multiple_files=True
        )
        process_button = st.button("🛠 Process PDFs")

    # 🔄 Process PDFs when the button is clicked
    if process_button and uploaded_files:
        with st.spinner('⚙️ Extracting & Analyzing PDFs...'):
            all_chunks = {}  # Thay đổi từ set sang dict để lưu trữ chunk và ID

            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)
                chunks = chunk_text(text)

                # Lọc các chunk trùng lặp dựa trên nội dung
                for chunk in chunks:
                    chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                    if chunk_id not in all_chunks:
                        all_chunks[chunk_id] = chunk

           # Convert non-duplicate chunks to Document Objects
            doc_chunks = convert_to_documents(list(all_chunks.values()))

            st.session_state.vector_db = VectorDB(doc_chunks)
            st.session_state.retriever = st.session_state.vector_db.get_retriever()

        with st.sidebar:
            st.markdown(
                f"""
                <div style='background-color: #2A2A2A; padding: 10px; border-radius: 8px; color: #50C878; text-align: center;'>
                    ✅ Successfully processed {len(uploaded_files)} PDF(s)!
                </div>
                """,
                unsafe_allow_html=True
            )

    # 🔍 Check if retriever is initialized
    if not st.session_state.retriever:
        st.warning("📌 Please upload and process PDFs before asking questions.")
        return

    # Layout: Chat Area (center) | Relevant Documents (right sidebar)
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

            seen = set()
            unique_docs = []
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

    # 📌 Sidebar on the right for retrieved relevant documents
    with col_relevant:
        st.markdown(
            """
            <h3 style='color: #E0E0E0; display: flex; align-items: center;'>
                <span style='margin-right: 8px;'>🔍</span> Relevant Documents
            </h3>
            """,
            unsafe_allow_html=True
        )

        if not st.session_state.relevant_docs:
            st.markdown(
                """
                <div style='background-color: #2A2A2A; padding: 15px; border-radius: 8px; text-align: center; color: #A0A0A0;'>
                    <span style='font-size: 16px;'>📄 No relevant documents found yet.</span><br>
                    <small style='color: #808080;'>Try asking a question to retrieve related content!</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            with st.container(height=500):
                for idx, doc in enumerate(st.session_state.relevant_docs):
                    with st.expander(f"{doc.page_content[:30]}", expanded=False):
                        st.markdown(
                            f"""
                            <div style='background-color: #2A2A2A; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                <p style='color: #B0B0B0; font-size: 14px; margin: 5px 0;'>
                                    {doc.page_content}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# Run the app
if __name__ == "__main__":
    main()
