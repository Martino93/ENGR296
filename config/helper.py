import streamlit as st


# Helper function to create columns
def create_columns(col1_content, col2_content):
    col1, col2 = st.columns(2)
    with col1:
        col1_content()
    with col2:
        col2_content()
