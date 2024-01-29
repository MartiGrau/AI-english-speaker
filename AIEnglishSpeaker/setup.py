import streamlit as st
from configuration.models import *
from dotenv import load_dotenv, find_dotenv


# -- SETUP -- #
@st.cache_resource(show_spinner=False)
def var_initialization():
    # Load environment variables
    loaded_success = load_dotenv(find_dotenv()) 
    if not loaded_success:
        raise ValueError('API KEY not loaded correctly')
    
    
    from agent.chat import load_llm
    from SST.model import load_sst_model

    # Load agent
    return {
        'chain': load_llm(),
        #'sst_model': load_sst_model(SST_TYPE),
        }
