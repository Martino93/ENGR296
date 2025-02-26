import os
import io
import re
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.chains.sql_database.query import create_sql_query_chain

from ai_agents.sql_agent.backend.core import ask_sql_agent
from langchain_core.runnables import chain


DB_PATH = "db/Chinook_Sqlite.sqlite"
DB = SQLDatabase.from_uri("sqlite:///" + DB_PATH)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def query_bot(llm_model="OpenAI"):
    st.header("SQL Builder Agent")
    st.write(
        "This agent can help you build SQL queries based on your Data Warehouse schema."
    )

    prompt = st.text_input(
        "Prompt", placeholder="Ask me anything, and I will help you build a SQL query."
    )

    if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
    ):
        st.session_state.chat_answers_history = []
        st.session_state.user_prompt_history = []
        st.session_state.chat_history = []

    if llm_model == "OpenAI":

        # run agent
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        toolkit = SQLDatabaseToolkit(db=DB, llm=model)
        sql_agent = create_sql_agent(toolkit=toolkit, llm=model, verbose=True)

    if prompt:
        with st.spinner("Thinking..."):
            buffer = io.StringIO()

            # redirect stdout
            old_stdout = sys.stdout
            sys.stdout = buffer

            # run agent
            response = sql_agent.invoke(prompt)

            # restore stdout
            sys.stdout = old_stdout

            reasoning_text = buffer.getvalue()
            reasoning_text = clean_reasoning_text(reasoning_text)

            st.write(response)
            st.code(reasoning_text, wrap_lines=True)


def db_size():
    size = os.path.getsize(DB_PATH)
    message = f"Database size: {size / (1024 * 1024):.2f} MB"
    return message


def display_db_info():

    size_message = db_size()

    st.sidebar.info(
        f"""
    Database Info
    
    Dialect:
    {DB.dialect}
    
    Usable tables:
    {','.join(i for i in DB.get_usable_table_names())}
    
    {size_message}
    """
    )


def clean_reasoning_text(text):

    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    return ansi_escape.sub("", text)


# def query():

#     query = "SELECT * FROM Employee LIMIT 5"

#     try:
#         response = DB.run(query)
#         print("Query Results:")
#         print(response)
#     except Exception as e:
#         print("Error executing query:", e)

#     # CHAINS
#     sql_chain = create_sql_query_chain(llm, DB)
#     sql_chain.get_prompts()[0].pretty_print()

#     question = "how many employees are there? You MUST RETURN ONLY MYSQL QUERIES."
#     response = sql_chain.invoke({"question": question})
#     print(response)


# @chain
# def get_correct_sql_query(input):
#     context = input["context"]
#     question = input["question"]

#     intruction = """
#         Use above context to fetch the correct SQL query for following question
#         {}

#         Do not enclose query in ```sql and do not write preamble and explanation.
#         You MUST return only single SQL query.
#     """.format(
#         question
#     )

#     response = ask_sql_agent(context=context, question=intruction)

#     return response
