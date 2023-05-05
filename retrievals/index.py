import os
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.document_loaders import TextLoader

class Index:
    def __init__(self, data_path: str, index_folder: str, create_index: bool = True) -> None:
        """
        Initialization function to create an Index object.

        Args:
        - data_path: str, path to the data file.
        - index_folder: str, path to the index folder.
        - create_index: bool, whether to create the index file. Default is True.
        """
        self.data_path: str = data_path
        self.index_folder: str = index_folder
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()

        # Check if the index file already exists
        self.is_indexed: bool = os.path.exists(os.path.join(index_folder, 'index.faiss'))
        
        # If create_index is True and the index file doesn't exist, initialize TextLoader, SpacyTextSplitter,
        # and OpenAIEmbeddings and create the index file.
        if create_index and not self.is_indexed:
            self.loader: TextLoader = TextLoader(data_path)
            self.text_splitter: SpacyTextSplitter = SpacyTextSplitter(chunk_size=256, pipeline="en_core_web_sm")
            self.create_index()

    def create_index(self) -> None:
        """
        Function to create the index by reading the data and generating the index file.
        """
        # Load the text data
        documents: List[str] = self.loader.load()
        # Split the text using SpacyTextSplitter
        texts: List[List[str]] = self.text_splitter.split_documents(documents)
        # Generate text embeddings using OpenAIEmbeddings and store the embeddings using FAISS
        faq_doc: FAISS = FAISS.from_documents(texts, self.embeddings)
        faq_doc.save_local(folder_path=self.index_folder)
        self.is_indexed: bool = True

if __name__ == "__main__":
    # Initialize the faq Index object
    faq_index = Index(data_path=os.path.join(os.getcwd(), 'data/faq.txt'), 
                      index_folder=os.path.join(os.getcwd(), 'indices/faq'))
