from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from util.printer import Printer
import time
import shutil
import os
import hashlib

class VectorDB:
    def __init__(self, chunks, persist_directory="./chroma_db"):
        self.printer = Printer()
        self.persist_directory = persist_directory
        start_time = time.time()
        
        self.printer.print(f"Initializing VectorDB with {len(chunks)} chunks", "cyan")
        
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
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
        
    def _create_vectordb(self, chunks):
        """Creates and initializes the vector database with unique IDs for each chunk."""
        start_time = time.time()
        self.printer.print("Creating vector database...", "yellow")
        
        # Generate unique IDs for each chunk based on its content
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

    def _reset_database(self):
        """Deletes the existing database directory and resets it."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.printer.print("🗑 Old database deleted!", "yellow")

    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 20}, similarity_threshold: float = 0.8):
        """
        Retrieves documents using similarity search and filters out those with similarity scores below the threshold.
        
        :param search_type: Type of search (default is "similarity").
        :param search_kwargs: Search parameters, e.g., the number of results to retrieve.
        :param similarity_threshold: Minimum similarity score to keep results.
        :return: A retriever that only returns documents meeting the similarity threshold.
        """
        self.printer.print(
            f"Getting retriever with search_type={search_type}, search_kwargs={search_kwargs}, similarity_threshold={similarity_threshold}", "cyan"
        )
        
        # Retrieve results and filter based on similarity scores
        def filtered_get_relevant_documents(query):
            docs_and_scores = self.vector_db.similarity_search_with_relevance_scores(query, **search_kwargs)
            return [doc for doc, score in docs_and_scores if score >= similarity_threshold]
        
        # Create retriever and override its method to apply filtering
        retriever = self.vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        retriever.get_relevant_documents = filtered_get_relevant_documents
        
        return retriever
