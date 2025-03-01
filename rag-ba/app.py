import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2

load_dotenv()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use the retrieved context information, not your internal knowledge."
    "If the context retrieval is not provided, say that the document is not provided.ftit"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    collection_name="example_collection",
)

retriever = vectorstore.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.set_page_config(layout="wide")
st.sidebar.title("Chatbot Sidebar")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

col1, col2 = st.columns([3, 1])

with col1:
    if uploaded_file is not None:
        # Open the PDF file and extract text
        reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()

        # Display the PDF content
        st.write(pdf_text)

# Column for displaying the chat messages
with col2:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    st.write("<br>", unsafe_allow_html=True)

    # User input and question answering
    user_input = st.text_input("Ask about the document...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        retrieved_docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        input_data = {
            "input": user_input,
            "context": context,
        }

        results = rag_chain.invoke(input_data)

        st.session_state.messages.append(
            {"role": "assistant", "content": results["answer"]}
        )
        st.chat_message("assistant").write(results["answer"])
