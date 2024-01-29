
import streamlit as st
# Recommit session state values
def update_session_state(session_state):
    try:
        for key in list(st.session_state.keys()):
            session_state[key] = session_state[key]
    except:
        pass
    return session_state