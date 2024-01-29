import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from .template import system_prompt_template

def load_llm():
    # Load LLM model
    llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0.0)

    # Conversation chain
    return LLMChain(
        prompt=system_prompt_template,
        llm=llm,
        verbose=True,
        #memory=ConversationBufferMemory(ai_prefix="AI Assistant")
    )

def add_to_chat_history(message, role, print_to_chat=True):
    st.session_state.messages.append({'role': role, 'content': message})
    if print_to_chat:
        st.chat_message(role).markdown(message)

def answering_question(session_state):
    print(list(session_state.keys()))
    with st.spinner('Answering...'):
        answer =  session_state.resources['chain'].predict( 
            history='\n'.join([message['content'] for message in session_state.messages]),
            input=session_state['question']
        )
    return answer