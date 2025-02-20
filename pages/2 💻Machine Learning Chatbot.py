import streamlit as st
from streamlit.components.v1 import html

from config.helper import create_columns


# get analytics contents from analytics.py
from analytics import analyze_heart_failure_data


st.set_page_config(layout="wide")


def header():
    st.title("Predictive Analytics")
    st.info(
        """Here we display relevant machine learning analytics.
            See the data dictionary for more information on the datasets used.
            """
    )

    # page divider
    st.write("---")


def display_ml_breast_cancer_model():

    cs235_page = "Models/rona_antonio_mohamed_martino_cs235_phase_1.html"

    with open(cs235_page, "r") as f:
        html_content = f.read()

    html(html_content, height=84000)


# PAGE CONTENTS

header()

create_columns(
    lambda: display_ml_breast_cancer_model(),
    lambda: st.image("assets/heart-pulse-svgrepo-com.svg", width=100),
)
