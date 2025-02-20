import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_NAME = os.environ["INDEX_NAME"]


def run_llm(query: str, chat_history=None):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    llm = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    # handle chat history
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    if not chat_history:
        result = qa.invoke({"input": query})
    else:
        result = qa.invoke({"input": query, "chat_history": chat_history})

    new_result = {
        "query": result.get("input"),
        "result": result.get("answer"),
        "source_doc": result.get("context"),
    }

    return new_result


if __name__ == "__main__":

    res = run_llm("What is my most recent weight?")
    print(res.get("result"))
