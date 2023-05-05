from typing import Optional
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def load_index(index_folder) -> Optional[FAISS]:
    """
    Loads the index from a previously created index file.

    Returns:
    - Optional[FAISS]: The loaded FAISS object, or None if loading failed.
    """
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(folder_path=index_folder, embeddings=embeddings)
    except Exception as e:
        print(f"Failed to load index: {e}")
        return None
