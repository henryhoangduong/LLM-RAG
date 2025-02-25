import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY")
)

st.title("Quickstart app")

prompt = st.chat_input("Say something...")  

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt:  
    st.session_state.messages.append({"role": "user", "content": prompt})  
    with st.chat_message("user"):  
        st.write(prompt)  

    time.sleep(1)  
    st.session_state.messages.append({"role": "bot", "content": prompt})  
    with st.chat_message("bot"):  
        st.write(prompt) 