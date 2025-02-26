import os
from dotenv import load_dotenv

load_dotenv()

import faiss

# Import necessary modules from LangChain
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent

from langchain_openai.chat_models.base import ChatOpenAI



OPENAI_KEY = os.environ["OPENAI_API_KEY"]


loader = UnstructuredHTMLLoader("Models/cs235_phase_1.html")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(documents=split_docs, embedding=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})


qa_chain = ChatOpenAI(
    llm=OpenAI(temperature=0),
    # chain_type="stuff",  # 'stuff' concatenates retrieved chunks into a single prompt
    retriever=retriever,
    return_source_documents=True,
)


qa_tool = Tool(
    name="NotebookQA",
    func=qa_chain,
    description="Answers questions about the exported Jupyter Notebook HTML file.",
)


agent = initialize_agent(
    tools=[qa_tool],
    llm=OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


question = "What is the main objective of this notebook?"
response = agent.invoke(question)
print("Response:", response)
