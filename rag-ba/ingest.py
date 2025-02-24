from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4


def chunk_text():
    pdf_path = "./docs/GIÁO-TRÌNH-PHÂN-TÍCH-DỮ-LIỆU-KINH-DOANH.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = chunk_text()
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )
    vector_store.add_documents(documents=chunks, id=uuids)
