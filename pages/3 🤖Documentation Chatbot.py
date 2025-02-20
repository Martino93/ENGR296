import streamlit as st

from ai_agents.health_chatbot.chatbot import instructions, chatbot, analytics

st.sidebar.info(
    """
        Here, you can chat with the in-house LLM agent pulling semantic data from a vector database.
        """
)


instructions()
analytics()
chatbot()
