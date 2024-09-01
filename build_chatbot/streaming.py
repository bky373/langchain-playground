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

model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
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

result = trimmer.invoke(messages)
print(result)
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)

config = {"configurable": {"session_id": "abc21"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm borahm. tell me a joke.")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
"""
|Hi| Bor|ah|m|!| Here|’s| a| joke| for| you|:

|Why| did| the| scare|crow| win| an| award|?

|Because| he| was| outstanding| in| his| field|!||
"""
