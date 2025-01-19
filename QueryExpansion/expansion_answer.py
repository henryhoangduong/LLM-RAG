from helper_utils import word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
reader = PdfReader("data/fastfacts-what-is-climate-change.pdf")
pdf_texts = [p.extract_text().strip() for p in reader]

pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n",". "," ",""], chunk_size = 1000, chunk_overlap = 0
)

character_splitter = character_splitter.split_text("\n\n".join(pdf_texts))