from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    collection_name="example_collection",
)

retriever = vectorstore.as_retriever()
