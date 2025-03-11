from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from util.printer import Printer
import time
import shutil
import os
import hashlib
import sys

# Workaround to ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class VectorDB:
    def __init__(self, chunks=None, persist_directory="./chroma_db"):
        self.printer = Printer()
        self.persist_directory = persist_directory
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # If the old database directory exists, delete it before creating a new one.
        if os.path.exists(self.persist_directory):
            self.printer.print("🗑 Removing old vector database...", "yellow")
            shutil.rmtree(self.persist_directory)

        # init vector_db
        if chunks:
            start_time = time.time()
            self.printer.print(f"Initializing VectorDB with {len(chunks)} chunks", "cyan")
            
            try:
                self.vector_db = self._create_vectordb(chunks)
            except ValueError as e:
                if "Could not connect to tenant" in str(e):
                    self.printer.print("⚠ Database error! Resetting and recreating...", "red")
                    self._reset_database()
                    self.vector_db = self._create_vectordb(chunks)
                else:
                    raise e
            
            elapsed_time = time.time() - start_time
            self.printer.print(f"Total initialization time: {elapsed_time:.2f} seconds", "bold_cyan")
        else:
            self.printer.print("No chunks provided. VectorDB will be empty.", "yellow")
            self.vector_db = None
        
    def _create_vectordb(self, chunks):
        start_time = time.time()
        self.printer.print("Creating vector database...", "yellow")
        
        # Generate unique IDs for each chunk based on content
        ids = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]
        
        vector_db = Chroma.from_documents(
            chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
            ids=ids
        )
        vector_db.persist() 
        
        elapsed_time = time.time() - start_time
        self.printer.print(f"✅ Vector database successfully created in {elapsed_time:.2f} seconds", "bold_green")
        
        return vector_db

    def add_documents(self, chunks):
        """
        Add new documents to existing vector database.
        If vector database doesn't exist, create a new one.
        """
        if not chunks:
            self.printer.print("No chunks to add", "yellow")
            return
            
        start_time = time.time()
        self.printer.print(f"Adding {len(chunks)} new chunks to vector database", "cyan")
        
        # Generate unique IDs for each chunk based on content
        ids = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]
        
        if self.vector_db is None:
            # If no vector database exists, create new one
            self.vector_db = self._create_vectordb(chunks)
        else:
            # If vector database exists, add new documents
            try:
                self.vector_db.add_documents(chunks, ids=ids)
                self.vector_db.persist()
                elapsed_time = time.time() - start_time
                self.printer.print(f"✅ Added {len(chunks)} chunks in {elapsed_time:.2f} seconds", "bold_green")
            except Exception as e:
                self.printer.print(f"❌ Error adding documents: {e}", "red")
                raise e

    def _reset_database(self):
        """Delete the old database if there are errors and recreate it."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.printer.print("🗑 Old database deleted!", "yellow")

    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 20}):
        """
        Get a retriever instance from the vector database.
        Args:
            search_type: Type of search to perform ('similarity' by default)
            search_kwargs: Additional search parameters (default k=20 for top results)
        Returns:
            Retriever instance or None if database is not available
        """
        if self.vector_db is None:
            self.printer.print("⚠ No vector database available", "red")
            return None
            
        self.printer.print(f"Getting retriever with search_type={search_type}, search_kwargs={search_kwargs}", "cyan")
        return self.vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
