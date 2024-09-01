import os
from operator import itemgetter

import dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    HumanMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

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

trimmer = trim_messages(
    max_tokens=56,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm borahm"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

result = trimmer.invoke(
    messages
)  # [SystemMessage(content="you're a good assistant"), HumanMessage(content='whats 2 + 2'), AIMessage(content='4'),
# HumanMessage(content='thanks'), AIMessage(content='no problem!'), HumanMessage(content='having fun?'),
# AIMessage(content='yes!')]

print(result)
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
# response = chain.invoke(
#     {
#         "messages": messages + [HumanMessage(content="What's my name?")],
#         "language": "Korean",
#     }
# )
# print(
#     response.content
# )  # 죄송하지만, 당신의 이름을 알 수 없습니다. 당신이 원하는 이름을 말씀해 주시면 그 이름으로 불러드릴 수 있습니다!
# response = chain.invoke(
#     {
#         "messages": messages + [HumanMessage(content="What math problems did I ask")],
#         "language": "English",
#     }
# )
# print(
#     response.content
# )  # You asked about the math problem "what's 2 + 2." If you have more math questions or any other topics you'd like
# # to discuss, feel free to ask!

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)

config = {"configurable": {"session_id": "abc20"}}
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="What's my name?")],
        "language": "English",
    },
    config=config,
)
print(response.content)  # I don't know your name. Can you tell me?

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="What math problem did I ask?")],
        "language": "English",
    },
    config=config,
)
print(
    response.content
)  # You haven't asked a math problem yet. If you have one in mind, feel free to share it!
