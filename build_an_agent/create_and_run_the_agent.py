import dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4")
tools = [(TavilySearchResults(max_results=2))]

agent_executor = create_react_agent(model, tools)
# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
# print(response["messages"])
# """[HumanMessage(content='hi!', id='75066df5-3cf9-48f4-a74a-f6d47bceb48d'), AIMessage(content='Hello! How can I
# assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10,
# 'prompt_tokens': 83, 'total_tokens': 93}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason':
# 'stop', 'logprobs': None}, id='run-07a397b3-b371-432c-ba5e-c988978dbc5b-0', usage_metadata={'input_tokens': 83,
# 'output_tokens': 10, 'total_tokens': 93})]"""

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="what's the weather in Suwon?")]}
)
print(response["messages"])
"""[HumanMessage(content="what's the weather in Suwon?", id='d87a21f5-304a-4640-933e-3b8484fb39ff'), 
AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_fKRwy7SimaRBecMQ9mI5MSqV', 'function': {
'arguments': '{\n  "query": "current weather in Suwon"\n}', 'name': 'tavily_search_results_json'}, 
'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 
89, 'total_tokens': 112}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 
'logprobs': None}, id='run-cbd9857f-bec2-4dad-a7e5-309e7070953a-0', tool_calls=[{'name': 
'tavily_search_results_json', 'args': {'query': 'current weather in Suwon'}, 'id': 'call_fKRwy7SimaRBecMQ9mI5MSqV', 
'type': 'tool_call'}], usage_metadata={'input_tokens': 89, 'output_tokens': 23, 'total_tokens': 112}), ToolMessage(
content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Suwon\', \'region\': \'\', 
\'country\': \'South Korea\', \'lat\': 36.45, \'lon\': 127.13, \'tz_id\': \'Asia/Seoul\', \'localtime_epoch\': 
1725204367, \'localtime\': \'2024-09-02 00:26\'}, \'current\': {\'last_updated_epoch\': 1725203700, \'last_updated\': 
\'2024-09-02 00:15\', \'temp_c\': 19.6, \'temp_f\': 67.3, \'is_day\': 0, \'condition\': {\'text\': \'Mist\', 
\'icon\': \'//cdn.weatherapi.com/weather/64x64/night/143.png\', \'code\': 1030}, \'wind_mph\': 2.2, \'wind_kph\': 
3.6, \'wind_degree\': 180, \'wind_dir\': \'S\', \'pressure_mb\': 1009.0, \'pressure_in\': 29.79, \'precip_mm\': 0.0, 
\'precip_in\': 0.0, \'humidity\': 94, \'cloud\': 22, \'feelslike_c\': 19.6, \'feelslike_f\': 67.3, \'windchill_c\': 
19.6, \'windchill_f\': 67.3, \'heatindex_c\': 19.6, \'heatindex_f\': 67.3, \'dewpoint_c\': 18.7, \'dewpoint_f\': 
65.6, \'vis_km\': 2.0, \'vis_miles\': 1.0, \'uv\': 1.0, \'gust_mph\': 2.8, \'gust_kph\': 4.5}}"}, 
{"url": "https://www.timeanddate.com/weather/south-korea/suwon/hourly", "content": "Hour-by-Hour Forecast for Suwon, 
South Korea. Currently: 74 \\u00b0F. Clear. (Weather station: Osan Ab, South Korea). See more current weather."}]', 
name='tavily_search_results_json', id='97c971d8-d833-4a87-9096-5695285b58e8', 
tool_call_id='call_fKRwy7SimaRBecMQ9mI5MSqV', artifact={'query': 'current weather in Suwon', 'follow_up_questions': 
None, 'answer': None, 'images': [], 'results': [{'title': 'Weather in Suwon', 'url': 'https://www.weatherapi.com/', 
'content': "{'location': {'name': 'Suwon', 'region': '', 'country': 'South Korea', 'lat': 36.45, 'lon': 127.13, 
'tz_id': 'Asia/Seoul', 'localtime_epoch': 1725204367, 'localtime': '2024-09-02 00:26'}, 'current': {
'last_updated_epoch': 1725203700, 'last_updated': '2024-09-02 00:15', 'temp_c': 19.6, 'temp_f': 67.3, 'is_day': 0, 
'condition': {'text': 'Mist', 'icon': '//cdn.weatherapi.com/weather/64x64/night/143.png', 'code': 1030}, 'wind_mph': 
2.2, 'wind_kph': 3.6, 'wind_degree': 180, 'wind_dir': 'S', 'pressure_mb': 1009.0, 'pressure_in': 29.79, 'precip_mm': 
0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 22, 'feelslike_c': 19.6, 'feelslike_f': 67.3, 'windchill_c': 19.6, 
'windchill_f': 67.3, 'heatindex_c': 19.6, 'heatindex_f': 67.3, 'dewpoint_c': 18.7, 'dewpoint_f': 65.6, 'vis_km': 2.0, 
'vis_miles': 1.0, 'uv': 1.0, 'gust_mph': 2.8, 'gust_kph': 4.5}}", 'score': 0.9995476, 'raw_content': None}, 
{'title': 'Hourly forecast for Suwon, South Korea - timeanddate.com', 'url': 
'https://www.timeanddate.com/weather/south-korea/suwon/hourly', 'content': 'Hour-by-Hour Forecast for Suwon, 
South Korea. Currently: 74 °F. Clear. (Weather station: Osan Ab, South Korea). See more current weather.', 
'score': 0.9932288, 'raw_content': None}], 'response_time': 2.86}), AIMessage(content='The current weather in Suwon, 
South Korea is misty with a temperature of 19.6°C (67.3°F). The wind is coming from the south at a speed of 3.6 kph (
2.2 mph). The humidity is at 94%, and visibility is 2 km (1 mile). The pressure is 1009.0 mb. [Source](
https://www.weatherapi.com/)', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {
'completion_tokens': 88, 'prompt_tokens': 592, 'total_tokens': 680}, 'model_name': 'gpt-4-0613', 
'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 
id='run-78255291-8c72-4352-b0fd-b8de94fcec0c-0', usage_metadata={'input_tokens': 592, 'output_tokens': 88, 
'total_tokens': 680})]
"""
