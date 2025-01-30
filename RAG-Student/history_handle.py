from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory

import yaml

with open("history_config.yml","r") as f:
    history_config = yaml.safe_load(f)

