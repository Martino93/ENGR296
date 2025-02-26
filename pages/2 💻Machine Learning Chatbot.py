import streamlit as st
from streamlit.components.v1 import html

from config.helper import create_columns

from ai_agents.ml_agent.ollama_agent import chatbot


st.set_page_config(layout="wide")


def header():
    st.title("Predictive Analytics")
    st.info(
        """Here we display relevant machine learning analytics.
            See the data dictionary for more information on the datasets used.
            
        Instructions
        
        Here we can ask questions to the chatbot which will then retrieve information from the text embeddings located in a local vector database.
        
        Sample questions:
        - what data mining techniques were used?
        - what dataset was used for this project?
        - what were the diagnosis Class Counts?
            """
    )

    # page divider
    st.write("---")


def display_ml_breast_cancer_model():

    cs235_page = "Models/cs235_phase_1.html"

    with open(cs235_page, "r") as f:
        html_content = f.read()

    st.html(html_content)


# PAGE CONTENTS

header()


create_columns(
    lambda: display_ml_breast_cancer_model(),
    lambda: chatbot(),
)
