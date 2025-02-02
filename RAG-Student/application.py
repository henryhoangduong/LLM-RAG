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

def load_history(history_name):
    history_id = get_history_id(history_name)
    if history_name != "New Session" and history_id != None:
        history = CustomHistory()
        history.load(history_id=history_id)
        return history
    return CustomHistory()

def Chatbot():
    # ====== Sessions & Neccessary Method ======
    if "send_input" not in st.session_state:
        st.session_state.send_input = False

    def clear_input():
        st.session_state.user_question = st.session_state.user_input
        st.session_state.user_input = ""
        
    def update_send_input_state():
        st.session_state.send_input = True
        clear_input()
        
    def history_choice_show(emp):
        typewriter_effect(emp, "Session hiện tại: " + st.session_state.history_choice, delay=0.00001)
        with emp:
            st.write(f"Session hiện tại: ```{st.session_state.history_choice}```")
        
    # ====== Header ======
    st.title('🤖 RAG Chatbot For Students')
    
    # ====== Sub sidebar ======
    if "number_of_documents" not in st.session_state:
        st.session_state.number_of_documents = 1
    with st.sidebar:
        # RAG Options
        rag_button = st.toggle("Cho phép Chatbot truy xuất Databases")
        if rag_button:
            st.sidebar.write("🟢 Lưu ý: Chế độ truy xuất Databases đang được bật")
            number_of_documents = st.sidebar.number_input(label="Số documents cung cấp", min_value=1, max_value=10)
            if st.session_state.number_of_documents != number_of_documents:
                clear_cache()
                st.session_state.number_of_documents = number_of_documents
            if st.session_state.rag_chat==False:
                rag_click()
        else:
            st.sidebar.write("🟠 Lưu ý: Chế độ truy xuất Databases đã tắt")
            if "rag_chat" in st.session_state:
                if st.session_state.rag_chat==True:
                    clear_cache()
            st.session_state.rag_chat = False
            
        # History Options
        list_chat_sessions = ["New Session"] + get_list_names()
        if "allow_change_history" not in st.session_state:
                st.session_state.allow_change_history = True     
        def history_choosing_change():
            st.session_state.allow_change_history = True
            
        history_choosing = st.sidebar.selectbox("Select a chat session", list_chat_sessions, key="chat_session", on_change=history_choosing_change)
        if st.session_state.allow_change_history:
            st.session_state.history_choice = history_choosing
            st.session_state.allow_change_history = False
        history_empty = st.empty()

    chain = load_chain()
    history = load_history(st.session_state.history_choice)
    history_choice_show(history_empty)
    # ====== Chat container ======
    chat_container = st.container()
    empty_generate = st.empty()
    user_input = st.text_input("Nhập câu hỏi của bạn tại đây!", key="user_input", on_change=update_send_input_state)
    
    # ====== Message History Show ======
    # Load again to avoid recording messages before
    history.history.clear()
    history = load_history(st.session_state.history_choice)
    
    with chat_container:
        for message in history.history.messages:
            st.chat_message(message.type).write(message.content)

    # ====== Chat ======
    if st.session_state.send_input:
        st.session_state.send_input = False
        memory_retrieve = get_chat_memory(history.history)
        print(memory_retrieve.load_memory_variables({}))
        with chat_container:
            st.chat_message('human').write(st.session_state.user_question)
            response = chain.run(st.session_state.user_question, memory_retrieve)
            history.add_a_conversation(st.session_state.user_question, response)
            typewriter_effect(empty_generate,response,delay=0.00005)
            st.chat_message('ai').write(response)
            if st.session_state.history_choice == "New Session":
                st.session_state.history_choice = history.history_name
                history_choice_show(history_empty)

            
        
            
    # print(test.messages)
    
    
    
if selected == "Chatbot":
    Chatbot()