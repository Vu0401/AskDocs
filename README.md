# AskDocs – Basic RAG for Document Q&A

**AskDocs** is a simple **Retrieval-Augmented Generation (RAG)** system designed for document-based question answering. Users can upload a PDF, ask questions, and receive AI-generated answers along with relevant document excerpts.

## ✨ Key Features  
- 📂 **Upload Documents** – Quickly process PDF files for search.  
- 🔍 **Information Retrieval** – Extracts relevant sections based on user queries.  
- 🧠 **AI-Powered Q&A** – Uses embeddings to enhance search results.  
- 🎨 **Minimal UI** – Built with Streamlit for a simple and interactive experience.  

## 🚀 How It Works  
1. **Upload a PDF file.**  
2. **Ask a question** related to the document.  
3. **The system retrieves** the most relevant text chunks.  
4. **View the answer** alongside supporting document excerpts.  

## 🔧 Technologies Used  
- **Python** – Core engine  
- **Streamlit** – User interface  
- **LangChain + ChromaDB** – Vector-based retrieval  
- **Hugging Face Embeddings** – Semantic search  

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
```

> ⚠️ **Important:** Before running the app, create a `.env` file in the project directory and add your **GEMINI_API_KEY**:  
> ```  
> GEMINI_API_KEY="your_api_key_here"  
> ```  
> This is required for AI-powered search to function properly.
