# AskDocs – AI-Powered RAG System for Document Q&A 📄🤖  

**AskDocs** is an advanced Retrieval-Augmented Generation (RAG) system designed to make document search and question-answering effortless. Simply upload PDFs, ask questions, and get AI-powered answers with relevant document excerpts displayed side by side.  

🔗 **Live Demo:** [AskDocs](https://askdocs.streamlit.app)  

## ✨ Key Features  
- 📂 **Upload & Search Instantly** – Drag and drop PDFs for quick processing.  
- 🧠 **AI-Enhanced Q&A** – Ask anything, and our intelligent model retrieves the most relevant answers.  
- 🔍 **Context-Aware Search** – Displays the most relevant document sections alongside the AI-generated response.  
- 📝 **Smart Summarization** – Extracts key insights for a concise, easy-to-read summary.  
- 📌 **Conversation History** – Keeps track of previous queries for seamless research.  
- 🎨 **User-Friendly Interface** – Clean, intuitive UI built with Streamlit.  

## 🚀 How It Works  
1. **Upload a PDF file.**  
2. **Ask a question** about the document’s content.  
3. Click **Search 🔍**, and the AI retrieves the best-matching response.  
4. **View the answer** along with the most relevant document excerpts.  

## 🔧 Technologies Used  
- **Python** (Core engine)  
- **Streamlit** (Interactive UI)  
- **LangChain + ChromaDB** (Vector-based document retrieval)  
- **Hugging Face Embeddings** (Semantic search & intelligent responses)  

## 🛠️ Local Installation  

```bash
# Clone the repository
git clone https://github.com/Vu0401/AskDocs.git

# Navigate to the project directory
cd AskDocs

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
