import random
import streamlit as st

from omegaconf import OmegaConf
from pathlib import Path
from dotenv import load_dotenv
from agent_utils import (
    load_initial_resources, answering_question, add_to_chat_history, delete_cache
)

def app(responses):
    st.title('AI-English-Speaker')

    # -- Initialize Chat -- #
    if 'messages' not in st.session_state:
        st.session_state.question = ''
        st.session_state.messages = []

    # -- Display chat messages from history -- #
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        st.session_state.welcome_response = random.choice(responses.welcome_responses)
        add_to_chat_history(st.session_state.welcome_response, 'assistant')

    if user_input := st.chat_input('Your text:'):
        # -- Add question to chat history -- #
        add_to_chat_history(user_input, 'user')

    if st.session_state.messages[-1]['role'] != 'assistant':
        # -- Answer the Questions -- #
        st.session_state.question = user_input
        answer = answering_question(st.session_state.question, responses.do_not_know_the_answer)
        add_to_chat_history(answer, 'assistant')


# -- MAIN -- #
def main():
    # -- Page Style -- #
    st.set_page_config(
        page_title="AI-English-Speaker",
        page_icon="https://github.com/MartiGrau/MartiGrau.github.io/blob/master/images/mgg_logo_black.png",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # -- Initializations -- #
    loaded_success = load_dotenv()
    if not loaded_success:
        raise ValueError('API KEY not loaded correctly')

    # -- Read Data -- #
    st.session_state.resources = load_initial_resources()
    responses = OmegaConf.load(str(Path(__file__).absolute().parent / 'prompts.yaml'))

    # -- SIDEBAR -- #
    #st.sidebar.title('AI-English-Speaker')
    #st.sidebar.write("")

    # -- Reload Cache -- #
    delete_cache_button = st.sidebar.button('Reiniciar Conocimiento')
    if delete_cache_button:
        delete_cache()

    # -- Sidebar Information -- #
    #st.sidebar.write('''Obtén más información en nuestra web: [StageInHome](https://stageinhome.com/).''')
    #st.sidebar.write("Para cualquier duda, [contacta](https://stageinhome.com/Contacto) con nosotros.")

    st.markdown('![](https://github.com/MartiGrau/MartiGrau.github.io/blob/master/images/mgg_logo_black.png)')    

    app(responses)

if __name__ == "__main__":
    main()