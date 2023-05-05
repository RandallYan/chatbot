import os
from typing import Optional

from langchain.chains import RetrievalQA
from tools.retrieval_qa import get_chain

index_folder: str = os.path.join(os.getcwd(), 'indices/faq')

def get_faq_chain(question: str = None,
                  index_folder: str = index_folder) -> Optional[RetrievalQA]:
    return get_chain(question=question, index_folder=index_folder)
