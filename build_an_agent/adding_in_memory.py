import dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4")
tools = [(TavilySearchResults(max_results=2))]
memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im borahm!")]}, config
):
    print(chunk)
    print("----")
"""{'agent': {'messages': [AIMessage(content='Hello Borahm! How can I assist you today?', additional_kwargs={
'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 87, 'total_tokens': 
100}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 
id='run-a0ae6cc9-d83d-4730-88ba-dd9d6bf6d724-0', usage_metadata={'input_tokens': 87, 'output_tokens': 13, 
'total_tokens': 100})]}} ----"""

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
"""
{'agent': {'messages': [AIMessage(content='Hello Borahm! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 87, 'total_tokens': 100}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-5497e640-fe85-4294-b033-96e561e806e7-0', usage_metadata={'input_tokens': 87, 'output_tokens': 13, 'total_tokens': 100})]}}
----
{'agent': {'messages': [AIMessage(content="Your name is Borahm, as you've just told me. How can I assist you further?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 112, 'total_tokens': 134}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-9cfa0ffd-3eb3-44c4-a414-9f705b4ae38d-0', usage_metadata={'input_tokens': 112, 'output_tokens': 22, 'total_tokens': 134})]}}
----
"""
config = {"configurable": {"thread_id": "xyz123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
"""
{'agent': {'messages': [AIMessage(content="Sorry, I can't assist with that.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 86, 'total_tokens': 96}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c617288c-444f-4b4e-ae8d-8d185d4e5c9c-0', usage_metadata={'input_tokens': 86, 'output_tokens': 10, 'total_tokens': 96})]}}
----
"""
