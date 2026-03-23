import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import Any

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Construct the path to the .env file
dotenv_path = os.path.join(project_root, ".env")

# Load environment variables from the specified .env file
print(f"Loading .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

# Global Pinecone client instance
_pinecone_client = None

# Global SentenceTransformer model instance
_embedding_model = None

def get_embedding_model():
    """Initializes and returns the SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading SentenceTransformer model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")
    return _embedding_model

def init_pinecone():
    """Initializes the Pinecone client and sets it as a global instance."""
    global _pinecone_client
    if _pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set in the .env file. Please check your .env file.")
        
        _pinecone_client = Pinecone(api_key=api_key)

def get_pinecone_index(index_name: str) -> Any:
    """Gets a Pinecone index.

    Args:
        index_name: The name of the index.

    Returns:
        The Pinecone index.
    """
    global _pinecone_client
    if _pinecone_client is None:
        init_pinecone() # Ensure client is initialized

    # Assuming 'all-MiniLM-L6-v2' produces 384-dimensional embeddings
    embedding_dimension = 384
    metric = "cosine" # Assuming cosine similarity

    index_list = _pinecone_client.list_indexes()
    index_names = [index.name for index in index_list]
    if index_name not in index_names:
        cloud = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not specified
        region = os.getenv("PINECONE_REGION", "us-west-2") # Default to us-west-2 if not specified

        _pinecone_client.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
    return _pinecone_client.Index(index_name)

def embed_text(text: str):
    """Embeds text using a sentence transformer model.

    Args:
        text: The text to embed.

    Returns:
        The embedding.
    """
    model = get_embedding_model()
    return model.encode(text)

def embed_texts(texts: list[str]):
    """Embeds a list of texts using a sentence transformer model.

    Args:
        texts: The list of texts to embed.

    Returns:
        The embeddings.
    """
    model = get_embedding_model()
    return model.encode(texts)

def upsert_vectors(index: Any, vectors: list):
    """Upserts vectors into the Pinecone index.

    Args:
        index: The Pinecone index.
        vectors: A list of tuples, where each tuple contains the id, vector, and metadata.
                 The metadata should be a dictionary containing the original text in a 'text' field.
                 Example:
                    [
                        ('id1', [0.1, 0.2, ...], {'text': 'This is the first sentence.'}),
                        ('id2', [0.3, 0.4, ...], {'text': 'This is the second sentence.'})
                    ]
    """
    index.upsert(vectors=vectors)

def query_pinecone(index: Any, query_vector, top_k: int = 5):
    """Queries the Pinecone index.

    Args:
        index: The Pinecone index.
        query_vector: The query vector.
        top_k: The number of results to return.

    Returns:
        The query results.
    """
    return index.query(vector=query_vector, top_k=top_k, include_metadata=True)

def rag(query: str):
    """
    Performs Retrieval-Augmented Generation using Pinecone.

    Args:
        query: The user's query.

    Returns:
        A tuple containing the retrieved context and the query.
    """
    init_pinecone() # Ensure client is initialized
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in the .env file.")
    
    index = get_pinecone_index(index_name)

    # Embed the query
    query_embedding = embed_text(query)

    # Query Pinecone
    query_results = query_pinecone(index, query_embedding.tolist())

    # For now, we'll just return the raw results.
    # In a real application, you would format this into a nice context string.
    return query_results, query
