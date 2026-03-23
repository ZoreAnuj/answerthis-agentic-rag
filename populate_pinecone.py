# Example script to upsert data into Pinecone (user would run this once)
from src.tools.pinecone_rag import init_pinecone, get_pinecone_index, embed_texts, upsert_vectors
import os
from docx import Document # Import the docx library

def read_word_document(file_path: str) -> list[str]:
    """
    Reads a Word document and extracts text from each paragraph.

    Args:
        file_path: The path to the Word document.

    Returns:
        A list of strings, where each string is a paragraph from the document.
    """
    document = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return paragraphs

def populate_pinecone_index(word_document_path: str, batch_size: int = 100):
    init_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in the .env file.")
    index = get_pinecone_index(index_name)

    # Read content from the Word document
    print(f"Reading document: {word_document_path}")
    paragraphs = read_word_document(word_document_path)
    print(f"Found {len(paragraphs)} paragraphs in the document.")

    total_upserted = 0
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}...")

        # Embed the batch of texts
        embeddings = embed_texts(batch)

        # Create vectors for upserting
        vectors_to_upsert = []
        for j, paragraph_text in enumerate(batch):
            # Generate a unique ID for each paragraph
            doc_id = f"doc_paragraph_{i+j}"
            vectors_to_upsert.append((doc_id, embeddings[j].tolist(), {"text": paragraph_text}))
        
        # Upsert the batch
        if vectors_to_upsert:
            upsert_vectors(index, vectors_to_upsert)
            total_upserted += len(vectors_to_upsert)
            print(f"Upserted {len(vectors_to_upsert)} vectors in this batch.")

    print(f"\nFinished upserting. Total paragraphs upserted: {total_upserted}")


if __name__ == "__main__":
    # The user needs to replace this with the actual path to their Word document
    word_doc_path = r"C:\Users\HP\Desktop\Agent_AI_Project\LONGEVITY CHATBOT Q&A KNOWLEDGE BASE .docx"
    populate_pinecone_index(word_doc_path)
