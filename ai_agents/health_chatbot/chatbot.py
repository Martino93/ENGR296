import os
import json
import streamlit as st
from streamlit_chat import message

from pinecone.grpc import PineconeGRPC as pgrpc

from ai_agents.health_chatbot.backend.core import run_llm


def instructions():
    st.markdown(
        """
        ## Instructions
        
        Here we can ask questions to the chatbot which will then retrieve information from the text embeddings located in a vector database.
        
        Sample questions:
        - What is the most recent weight?
        - What is the most recent blood pressure?
        - What is the most recent glucose level?
        - What is the name of the patient?
        - What type of information does Fitbit track?
        """
    )


def analytics():
    api_key = os.environ.get("PINECONE_API_KEY")
    pc = pgrpc(api_key=api_key)

    index_name = os.environ.get("INDEX_NAME")
    index = pc.Index(index_name)

    json_data = index.describe_index_stats()

    st.markdown("## Pinecone Index Stats")
    st.code(json_data, language="json")


def chatbot():
    st.header("LangChain Chatbot")

    prompt = st.text_input("Prompt", placeholder="Ask me anything...")

    if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
    ):
        st.session_state.chat_answers_history = []
        st.session_state.user_prompt_history = []
        st.session_state.chat_history = []

    if prompt:
        with st.spinner("Thinking..."):
            response = run_llm(query=prompt, chat_history=st.session_state.chat_history)
            sources = set(
                [doc.metadata.get("source") for doc in response.get("source_doc")]
            )

            formatted_response = (
                f"Response: {response.get('result')} \n\n Sources: {sources}"
            )

            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(formatted_response)
            st.session_state.chat_history.append(("human", prompt))
            st.session_state.chat_history.append(("ai", response.get("result")))

    if st.session_state.chat_answers_history:
        for i, (prompt, response) in enumerate(
            zip(
                st.session_state.user_prompt_history,
                st.session_state.chat_answers_history,
            )
        ):
            message(prompt, is_user=True, key=f"user_{i}")
            message(response, key=f"response_{i}")


# instructions()
# sidebar_analytics()
# chatbot()
