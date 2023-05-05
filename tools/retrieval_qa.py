from typing import Optional
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from tools.utils import load_index

def get_chain(question: str ,  # question: str,  # TODO: Change this to question: str = None
            index_folder: str = None,
            llm: ChatOpenAI = ChatOpenAI(temperature=0),
            verbose: bool = True) -> Optional[RetrievalQA]:
    """
    Creates and returns a RetrievalQA object based on a loaded FAISS index and a ChatOpenAI LLM.

    Args:
    - index_folder (str): The folder path for the FAISS index.
    - llm (ChatOpenAI): The ChatOpenAI language model to use.
    - verbose (bool): Whether to print information during creation.

    Returns:
    - Optional[RetrievalQA]: The created RetrievalQA object, or None if the index failed to load.
    """
    if question is None:
        raise ValueError("The 'question' parameter cannot be None.")
    if index_folder is None:
        raise ValueError("The 'index_folder' parameter cannot be None.")

    docsearch = load_index(index_folder)
    if docsearch is None:
        return None
    faq_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), verbose=verbose)
    return faq_chain.run(question)
