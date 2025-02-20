import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")

vault_path = "C:/Users/Hamed/Documents/ENGR296 Vault"

loader = PyPDFDirectoryLoader(vault_path)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(documents)

PineconeVectorStore.from_documents(
    documents, EMBEDDINGS, index_name=os.environ["INDEX_NAME"]
)
