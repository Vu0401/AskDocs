__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from rag.vectordb import VectorDB
from rag.rag import RAG
from util.util import extract_text_from_pdf, chunk_text, convert_to_documents

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
        st.session_state.rag = RAG(model="gemini/gemini-2.0-flash", functions=[])
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
        st.sidebar.title("🔍 AskDocs")
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
            all_chunks = set()  

            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)
                chunks = chunk_text(text)

                unique_chunks = [chunk for chunk in chunks if chunk not in all_chunks]
                all_chunks.update(unique_chunks)  

                st.session_state.pdf_chunks.append({
                    'filename': pdf_file.name,
                    'chunks': unique_chunks
                })

            doc_chunks = convert_to_documents(list(all_chunks))  
            
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
        # Thanh chat input cố định ở trên cùng
        question = st.chat_input("💭 Type your question here...", key="chat_input_top")

        # Container cuộn cho lịch sử chat
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                st.markdown("<br>", unsafe_allow_html=True)

        # Xử lý câu hỏi khi người dùng nhập
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with chat_container:
                with st.chat_message("user"):
                    st.write(question)

            relevant_docs = st.session_state.retriever.invoke(question)
            st.session_state.relevant_docs = list(set(relevant_docs))
            context = "\n".join([doc.page_content for doc in relevant_docs])

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
