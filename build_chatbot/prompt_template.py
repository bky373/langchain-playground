import os

import dotenv
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ],
)
chain = prompt | model
# response = chain.invoke({"messages": [HumanMessage(content="Hi!, I'm Borahm")]})
# print(response.content)  # Hi Borahm! How can I assist you today?

# with_message_history = RunnableWithMessageHistory(chain, get_session_history)
# config = {"configurable": {"session_id": "abc5"}}
# response = with_message_history.invoke(
#     [HumanMessage(content="Hi, I'm Br")],
#     config=config,
# )
# print(response.content)
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")], config=config
# )

# response = chain.invoke(
#     {
#         "messages": [HumanMessage(content="hi! I'm Br")],
#         "language": "Spanish",
#     }
# )
# print(response.content)  # ¡Hola, Br! ¿Cómo puedo ayudarte hoy?

with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)
config = {"configurable": {"session_id": "abc10"}}
response = with_message_history.invoke(
    {
        "messages": [
            HumanMessage(content="Hi, I'm Borahm"),
        ],
        "language": "Italian",
    },
    config=config,
)
print(response.content)  # Ciao! Come posso aiutarti oggi?

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="What's my name?")], "language": "Korean"},
    config=config,
)
print(response.content)  # 당신의 이름은 보람입니다.
