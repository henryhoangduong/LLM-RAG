import streamlit as st
from streamlit_option_menu import option_menu
import time
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# from utils import typewriter_effect
# from history_handle import (
#     CustomHistory,
#     get_list_names,
#     get_history_id,
#     get_chat_memory,
# )
import llm_chain

# from vectordb import (
#     get_list_documents,
#     get_document,
#     delete_document,
#     get_details,
#     create_vectordb_with_file,
# )

import yaml
import os


with open("db_config.yml", "r") as f:
    db_config = yaml.safe_load(f)

st.set_page_config(
    page_title="RAG Chatbot For Students",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="auto",
)

with st.sidebar:
    selected = option_menu(
        "Chatbot Options",
        ["Chatbot", "Databases"],
        icons=["chat", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"font-family": "Monospace"},
            "icon": {"color": "#71738d"},
            "nav-link": {"--hover-color": "#d2cdfa", "font-family": "Monospace"},
            "nav-link-selected": {
                "font-family": "Monospace",
                "background-color": "#a9a9ff",
            },
        },
    )


def clear_cache():
    st.cache_resource.clear()


def rag_click():
    st.session_state.rag_chat = True
    clear_cache()


@st.cache_resource
def load_chain():
    if st.session_state.rag_chat:
        print(
            f"====== Loading RAG Chat! (number of documents: {st.session_state.number_of_documents}) ======"
        )
        return llm_chain.load_rag_chain(st.session_state.number_of_documents)
    print("====== Loading normal chat! ======")
    return llm_chain.load_normal_chain()
