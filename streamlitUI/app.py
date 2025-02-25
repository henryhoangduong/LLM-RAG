import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
load_dotenv()
import os

st.title("Quickstart app")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Say something...")  

if prompt:  
    st.session_state.messages.append({"role": "user", "content": prompt})  
    with st.chat_message("user"):  
        st.write(prompt)  

    time.sleep(1)  
    bot_response = f"You said: {prompt}"  
    st.session_state.messages.append({"role": "bot", "content": bot_response})  
    with st.chat_message("bot"):  
        st.write(bot_response) 