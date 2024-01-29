'''Streamlit Demo App for AI English Speaker
This script is used to run a demo app for AI English Speaker using Streamlit.

> streamlit run run_demo.py
'''
import numpy as np
import streamlit as st
from agent.chat import add_to_chat_history, answering_question
from configuration.models import *
from utils import update_session_state
from setup import var_initialization

def web_info():
    # -- Page Style -- #
    st.set_page_config(
        page_title="English Speaker",
        page_icon="https://stageinhome.com/images/Logo.svg",
        layout="wide",
        initial_sidebar_state="auto",
    )
    
    # -- SIDEBAR -- #
    st.sidebar.title('English Speaker')
    st.sidebar.write("This app is a demo for AI English Speaker.")

    # -- MAIN PAGE -- #
    st.title("AI English Speaker")

    st.info(""" 
            1. Select a field of interest to talk about.
            2. Click the button to start recording.
            3. Click the button again to stop recording.
            """)
    
def app(cfg):
    # -- Initialize chatbot -- #
    if 'messages' not in st.session_state:
        st.session_state.question = ''
        st.session_state['messages'] = []

    # -- Display chat messages from history -- #
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if not st.session_state.messages:
        st.session_state.welcome_response = "Hi! What do you want to do?"
        add_to_chat_history(st.session_state.welcome_response, 'assistant')

    if user_input := st.chat_input('Your message'):
        # -- Add question to chat history -- #
        add_to_chat_history(user_input, 'user')

    if st.session_state.messages[-1]['role'] != 'assistant':
        st.session_state.user_input = user_input
        answer = answering_question(st.session_state)
        add_to_chat_history(answer, 'assistant')

    # Recommit session state values
    st.session_state = update_session_state(st.session_state)

    from SST.model import WhisperSTT
    text=WhisperSTT(language='en') 
    if text:
        st.write(text)

if __name__ == "__main__":
    # Load configuration
    cfg = None

    # Print web info
    web_info()

    # Initialize variables / models
    st.session_state.resources = var_initialization()

    # -- Initialize chatbot -- #
    app(cfg)