from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from docx import Document
from langchain_core.documents import Document as LC_Document
from pypdf import PdfReader
import torch
import os
import yaml


with open("db_config.yml","r") as f:
    db_config = yaml.safe_load(f)

    