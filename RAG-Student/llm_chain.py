from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
import vectordb

from operator import itemgetter
import yaml
import json

with open("db_config.yml", "r") as f:
    db_config = yaml.safe_load(f)

with open("model_config.yml", "r") as f:
    model_config = yaml.safe_load(f)

with open("history_config.yml", "r") as f:
    history_config = yaml.safe_load(f)


from prompt_template import memory_prompt_template, rag_prompt_template


def load_llm_model(
    model_path=model_config["chatbot_path"]["Q5bit"],
    model_type=model_config["chatbot_type"],
    model_config=model_config["chatbot_config"],
):
    llm_model = CTransformers(
        model=model_path, model_type=model_type, model_config=model_config
    )
    return llm_model


def get_PromptTemplate(template):
    return PromptTemplate.from_template(template)


def get_llm_chain(llm_model, prompt_template):
    pipeline = {
        "human_input": itemgetter("human_input"),
        "history": itemgetter("history"),
    }

class ChatChain:
    def __init__(self):
        pass

    def run(self):
        pass

def load_normal_chain():
    pass

def get_retriever_chain():
    pass


def get_retreiver_chain_pipeline():
    pass

class RAG_ChatChain:
    def __init__(self):
        pass 

    def run(self, question, memory):
        pass