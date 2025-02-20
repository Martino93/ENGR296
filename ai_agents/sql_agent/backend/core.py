### Question Answering using LLM
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser


def ask_sql_agent(question, llm_model):

    if llm_model == "llama3":
        base_url = "http://localhost:11434"
        model = "llama3"

        llm = ChatOllama(base_url=base_url, model=model)

        system = SystemMessagePromptTemplate.from_template(
            """You are helpful AI assistant who answer user question based on the provided context."""
        )

        prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".

        ### Question:
        {question}

        ### Answer:"""

        prompt = HumanMessagePromptTemplate.from_template(prompt)

        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        qna_chain = template | llm | StrOutputParser()

    return qna_chain.invoke({"question": question})
