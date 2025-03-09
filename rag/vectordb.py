from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from util.printer import Printer
import time
import shutil
import os

class VectorDB:
    def __init__(self, chunks, persist_directory="./chroma_db"):
        self.printer = Printer()
        self.persist_directory = persist_directory
        start_time = time.time()
        
        self.printer.print(f"Initializing VectorDB with {len(chunks)} chunks", "cyan")
        
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Kiểm tra lỗi tenant database
        try:
            self.vector_db = self._create_vectordb(chunks)
        except ValueError as e:
            if "Could not connect to tenant" in str(e):
                self.printer.print("⚠ Database bị lỗi! Đang xóa và tạo lại...", "red")
                self._reset_database()
                self.vector_db = self._create_vectordb(chunks)
            else:
                raise e
        
        elapsed_time = time.time() - start_time
        self.printer.print(f"Total initialization time: {elapsed_time:.2f} seconds", "bold_cyan")
        
    def _create_vectordb(self, chunks):
        start_time = time.time()
        self.printer.print("Creating vector database...", "yellow")
        
        # Khởi tạo database với persist_directory
        vector_db = Chroma.from_documents(
            chunks, 
            embedding=self.embedding, 
            persist_directory=self.persist_directory
        )
        vector_db.persist()  # Lưu database
        
        elapsed_time = time.time() - start_time
        self.printer.print(f"✅ Vector database created successfully in {elapsed_time:.2f} seconds", "bold_green")
        
        return vector_db

    def _reset_database(self):
        """Xóa database cũ nếu có lỗi và tạo lại từ đầu."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.printer.print("🗑 Đã xóa database cũ!", "yellow")

    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 20}):
        self.printer.print(f"Getting retriever with search_type={search_type}, search_kwargs={search_kwargs}", "cyan")
        return self.vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
