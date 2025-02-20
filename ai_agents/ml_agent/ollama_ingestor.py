import os
from dotenv import load_dotenv

load_dotenv()

import faiss

# Import necessary modules from LangChain
# from langchain.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_text_splitters.markdown import MarkdownTextSplitter


def ingest_notebook(dir_path, db_name):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )

    # loader = UnstructuredMarkdownLoader("Models/cs235_phase_2.md")
    loader = DirectoryLoader(
        dir_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )

    documents = loader.load()

    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    split_docs = text_splitter.split_documents(documents)

    # create index
    sample_vector = embeddings.embed_query("Hello, world!")
    index = faiss.IndexFlatL2(len(sample_vector))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    ids = vector_store.add_documents(documents=split_docs)

    # question = "What is the purpose of this document?"
    # ans = vector_store.search(query=question, k=5, search_type="similarity")
    # print(ans)

    # store vector_store in a file
    vector_store.save_local(db_name)


if __name__ == "__main__":

    dir_path = "Models/"
    db_name = "db/FAISS/cs235_project"
    ingest_notebook(dir_path, db_name)
