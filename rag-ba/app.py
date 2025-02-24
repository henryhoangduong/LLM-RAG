import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# System prompt for the assistant
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Initialize the prompt and LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize vectorstore for retrieval
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    collection_name="example_collection",
)

# Initialize retriever and RAG chain
retriever = vectorstore.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI setup
st.sidebar.title("Chatbot Sidebar")
st.sidebar.write("You can customize your chatbot settings here.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.write("<br>", unsafe_allow_html=True)

# User input
user_input = st.text_input("Say something...")

if user_input:
    # Store the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Use the RAG chain to retrieve relevant documents and generate an answer
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    input_data = {
        "input": user_input,
        "context": context,
    }

    # Generate the answer using the RAG chain
    results = rag_chain.invoke(input_data)

    # Store and display the assistant's response
    st.session_state.messages.append(
        {"role": "assistant", "content": results["answer"]}
    )
    st.chat_message("assistant").write(results["answer"])
