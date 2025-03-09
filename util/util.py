import PyPDF2
import re
from langchain.schema import Document
from util.printer import Printer

printer = Printer()

def convert_to_documents(chunks):
    """Convert text chunks into Document objects."""
    printer.print(f"Converting {len(chunks)} text chunks into Document objects...", "cyan")
    documents = [Document(page_content=chunk) for chunk in chunks]
    printer.print("Conversion completed successfully.", "bold_green")
    return documents

def extract_text_from_pdf(pdf_file):
    """Read a PDF file and extract text."""
    printer.print(f"Extracting text from PDF: {pdf_file.name}", "yellow")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    printer.print(f"PDF contains {len(pdf_reader.pages)} pages.", "cyan")
    for i, page in enumerate(pdf_reader.pages, 1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text

    printer.print(f"Total pages processed: {len(pdf_reader.pages)}", "bold_green")
    return text

def chunk_text(text, chunk_size=500):
    """Split text into chunks based on sentence boundaries or line breaks."""
    printer.print(f"Chunking text with size limit: {chunk_size} characters...", "yellow")
    printer.print(f"Initial text length: {len(text)} characters.", "cyan")

    # Split based on ".", "?", "!", or double line breaks "\n\n"
    chunks = re.split(r'(?<=[.?!])\s+|\n\n', text)
    printer.print(f"Initial split resulted in {len(chunks)} chunks.", "cyan")

    # Merge short chunks
    merged_chunks = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.replace("\n", " ").strip()  # Replace single line breaks with spaces
        if not chunk:  # Skip empty chunks
            continue

        if len(buffer) + len(chunk) < chunk_size:
            buffer += " " + chunk if buffer else chunk
        else:
            if buffer:
                merged_chunks.append(buffer.strip())
            buffer = chunk  # Start a new chunk

    if buffer:  # Add the remaining text
        merged_chunks.append(buffer.strip())

    printer.print(f"Total chunks created: {len(merged_chunks)}", "bold_green")
    return merged_chunks
