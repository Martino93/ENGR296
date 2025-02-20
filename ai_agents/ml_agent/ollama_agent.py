import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.html import UnstructuredHTMLLoader

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

vector_db_name = "db/FAISS/cs235_project"

vector_store = FAISS.load_local(
    vector_db_name, embeddings=embeddings, allow_dangerous_deserialization=True
)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "what was the highest F1 score obtained in Phase 1 of the project?"

response = rag_chain.invoke(question)
print(response)
