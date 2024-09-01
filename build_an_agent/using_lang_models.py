import dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4")
response = model.invoke([HumanMessage(content="hi!")])
print(response.content)  # 'Hi there!'

search = TavilySearchResults(max_results=2)
tools = [search]

model_with_tools = model.bind_tools(tools)

# response = model_with_tools.invoke(([HumanMessage(content="hi! I'm borahm")]))
# print(
#     f"response.content: {response.content}"
# )  # Hello Borahm! How can I assist you today?
# print(f"response.tool_calls: {response.tool_calls}")  # []

response = model_with_tools.invoke([HumanMessage(content="what's the weather in SF?")])
print(f"response.content: {response.content}")  # response.content:
print(f"response.tool_calls: {response.tool_calls}")
"""response.tool_calls: [{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San 
Francisco'}, 'id': 'call_gQbUg2BXn7gvZ5vEDXsHBCFn', 'type': 'tool_call'}]"""
