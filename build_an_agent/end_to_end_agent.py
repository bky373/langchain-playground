# Import relevant functionality
import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")

"""
{'agent': {'messages': [AIMessage(content="Hello Bob! Since you didn't ask a specific question, I don't need to use any tools to respond. It's nice to meet you. San Francisco is a wonderful city with lots to see and do. I hope you're enjoying living there. Please let me know if you have any other questions!", response_metadata={'id': 'msg_01Mmfzfs9m4XMgVzsCZYMWqH', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 271, 'output_tokens': 65}}, id='run-44c57f9c-a637-4888-b7d9-6d985031ae48-0', usage_metadata={'input_tokens': 271, 'output_tokens': 65, 'total_tokens': 336})]}}
----
{'agent': {'messages': [AIMessage(content=[{'text': 'To get current weather information for your location in San Francisco, let me invoke the search tool:', 'type': 'text'}, {'id': 'toolu_01BGEyQaSz3pTq8RwUUHSRoo', 'input': {'query': 'san francisco weather'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], response_metadata={'id': 'msg_013AVSVsRLKYZjduLpJBY4us', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 347, 'output_tokens': 80}}, id='run-de7923b6-5ee2-4ebe-bd95-5aed4933d0e3-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'san francisco weather'}, 'id': 'toolu_01BGEyQaSz3pTq8RwUUHSRoo'}], usage_metadata={'input_tokens': 347, 'output_tokens': 80, 'total_tokens': 427})]}}
----
{'tools': {'messages': [ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1717238643, \'localtime\': \'2024-06-01 3:44\'}, \'current\': {\'last_updated_epoch\': 1717237800, \'last_updated\': \'2024-06-01 03:30\', \'temp_c\': 12.0, \'temp_f\': 53.6, \'is_day\': 0, \'condition\': {\'text\': \'Mist\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/143.png\', \'code\': 1030}, \'wind_mph\': 5.6, \'wind_kph\': 9.0, \'wind_degree\': 310, \'wind_dir\': \'NW\', \'pressure_mb\': 1013.0, \'pressure_in\': 29.92, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 88, \'cloud\': 100, \'feelslike_c\': 10.5, \'feelslike_f\': 50.8, \'windchill_c\': 9.3, \'windchill_f\': 48.7, \'heatindex_c\': 11.1, \'heatindex_f\': 51.9, \'dewpoint_c\': 8.8, \'dewpoint_f\': 47.8, \'vis_km\': 6.4, \'vis_miles\': 3.0, \'uv\': 1.0, \'gust_mph\': 12.5, \'gust_kph\': 20.1}}"}, {"url": "https://www.timeanddate.com/weather/usa/san-francisco/historic", "content": "Past Weather in San Francisco, California, USA \\u2014 Yesterday and Last 2 Weeks. Time/General. Weather. Time Zone. DST Changes. Sun & Moon. Weather Today Weather Hourly 14 Day Forecast Yesterday/Past Weather Climate (Averages) Currently: 68 \\u00b0F. Passing clouds."}]', name='tavily_search_results_json', tool_call_id='toolu_01BGEyQaSz3pTq8RwUUHSRoo')]}}
----
{'agent': {'messages': [AIMessage(content='Based on the search results, the current weather in San Francisco is:\n\nTemperature: 53.6°F (12°C)\nConditions: Misty\nWind: 5.6 mph (9 kph) from the Northwest\nHumidity: 88%\nCloud Cover: 100% \n\nThe results provide detailed information like wind chill, heat index, visibility and more. It looks like a typical cool, foggy morning in San Francisco. Let me know if you need any other details about the weather where you live!', response_metadata={'id': 'msg_019WGLbaojuNdbCnqac7zaGW', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 1035, 'output_tokens': 120}}, id='run-1bb68bf3-b212-4ef4-8a31-10c830421c78-0', usage_metadata={'input_tokens': 1035, 'output_tokens': 120, 'total_tokens': 1155})]}}
----
"""
