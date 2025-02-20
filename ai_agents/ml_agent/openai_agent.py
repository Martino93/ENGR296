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


# Set your OpenAI API key
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

# Step 1: Load the exported HTML file
# Replace 'notebook_export.html' with the path to your exported HTML file.
loader = UnstructuredHTMLLoader("Models/cs235_phase_1.html")
documents = loader.load()

# Step 2: Split the document into smaller chunks
# This helps manage token usage and improves retrieval efficiency.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

# Step 3: Create a vector store from your document chunks using OpenAI embeddings
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(documents=split_docs, embedding=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Step 4: Create a RetrievalQA chain that uses the retriever and an LLM to answer questions
qa_chain = ChatOpenAI(
    llm=OpenAI(temperature=0),
    # chain_type="stuff",  # 'stuff' concatenates retrieved chunks into a single prompt
    retriever=retriever,
    return_source_documents=True,  # Optional: shows which documents were used
)

# Step 5: Wrap the QA chain as a tool for an agent
qa_tool = Tool(
    name="NotebookQA",
    func=qa_chain,
    description="Answers questions about the exported Jupyter Notebook HTML file.",
)

# Step 6: Initialize the agent with the QA tool
agent = initialize_agent(
    tools=[qa_tool],
    llm=OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Step 7: Ask a question
question = "What is the main objective of this notebook?"
response = agent.invoke(question)
print("Response:", response)
