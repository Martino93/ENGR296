import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)


def oxygen_saturation_chatbot():
    dir_path = "db/takeout-20241220T023650Z-001/Fitbit/Oxygen Saturation (SpO2)"

    tables = [
        i.path
        for i in os.scandir(dir_path)
        if "Daily" in i.path and i.path.endswith(".csv")
    ]

    df = pd.DataFrame()
    for table in tables:
        df = pd.concat([df, pd.read_csv(table)])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True
    )

    question = "what was the highest SpO2 value?"
    response = agent.invoke(question)
    return response


def active_zone_minutes_chatbot():
    dir_path = "db/takeout-20241220T023650Z-001/Fitbit/Active Zone Minutes (AZM)"

    tables = [i.path for i in os.scandir(dir_path) if i.path.endswith(".csv")]

    df = pd.DataFrame()
    for table in tables:
        df = pd.concat([df, pd.read_csv(table)])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True
    )

    question = "Give me a count of the different active zone minutes per day."
    response = agent.invoke(question)
    return response


active_zone_minutes_chatbot()
