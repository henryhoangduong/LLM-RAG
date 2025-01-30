import streamlit as st
from streamlit_option_menu import option_menu
import time
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


from utils import typewriter_effect
from history_handle import (
    CustomHistory,
    get_list_names,
    get_history_id,
    get_chat_memory,
)
import llm_chain
from vectordb import (
    get_list_documents,
    get_document,
    delete_document,
    get_details,
    create_vectordb_with_file,
)

import yaml
import os


with open("db_config.yml","r") as f:
    db_config = yaml.safe_load(f)

st.set_page_config(
    page_title="RAG Chatbot For Students",  
    page_icon=":robot_face:",  
    layout="centered", 
    initial_sidebar_state="auto" 
)