import os

import dotenv
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

model = ChatOpenAI(model="gpt-4o-mini")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi, I'm Borahm")], config=config
)
print(response.content)  # Hi Borahm! How can I assist you today?

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")], config=config
)
print(response.content)  # Your name is Borahm. How can I help you today?

config = {"configurable": {"session_id": "abc3"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")], config=config
)
print(
    response.content
)  # I'm sorry, but I don't know your name. How can I assist you today?

config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")], config=config
)
print(
    response.content
)  # Your name is Borahm. Is there anything specific you would like to talk about?
