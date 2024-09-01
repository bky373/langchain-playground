import dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4")
tools = [(TavilySearchResults(max_results=2))]

agent_executor = create_react_agent(model, tools)


async def streaming_tokens():
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content="whats the weather in Florida?")]},
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")


if __name__ == "__main__":
    import asyncio

    asyncio.run(streaming_tokens())
"""
--
Starting tool: tavily_search_results_json with inputs: {'query': 'current weather in Florida'}
Done tool: tavily_search_results_json
Tool output was: content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Floridablanca\', \'region\': \'Santander\', \'country\': \'Colombia\', \'lat\': 7.06, \'lon\': -73.09, \'tz_id\': \'America/Bogota\', \'localtime_epoch\': 1725205217, \'localtime\': \'2024-09-01 10:40\'}, \'current\': {\'last_updated_epoch\': 1725204600, \'last_updated\': \'2024-09-01 10:30\', \'temp_c\': 23.2, \'temp_f\': 73.8, \'is_day\': 1, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/116.png\', \'code\': 1003}, \'wind_mph\': 10.5, \'wind_kph\': 16.9, \'wind_degree\': 150, \'wind_dir\': \'SSE\', \'pressure_mb\': 1018.0, \'pressure_in\': 30.06, \'precip_mm\': 0.54, \'precip_in\': 0.02, \'humidity\': 73, \'cloud\': 25, \'feelslike_c\': 25.2, \'feelslike_f\': 77.4, \'windchill_c\': 22.2, \'windchill_f\': 72.0, \'heatindex_c\': 24.6, \'heatindex_f\': 76.3, \'dewpoint_c\': 18.8, \'dewpoint_f\': 65.8, \'vis_km\': 10.0, \'vis_miles\': 6.0, \'uv\': 5.0, \'gust_mph\': 15.0, \'gust_kph\': 24.1}}"}, {"url": "https://www.timeanddate.com/weather/usa/orlando/historic?month=1&year=2024", "content": "Weather reports from January 2024 in Orlando, Florida, USA with highs and lows. Sign in. News. News Home; Astronomy News; ... See more current weather. ... January 2024 Weather in Orlando \\u2014 Graph \\u00b0F. See Hour-by-hour Forecast for upcoming weather."}]' name='tavily_search_results_json' tool_call_id='call_9DdBKfWnQ4SQ0jphBIAUAuBW' artifact={'query': 'current weather in Florida', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Weather in Florida', 'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'Floridablanca', 'region': 'Santander', 'country': 'Colombia', 'lat': 7.06, 'lon': -73.09, 'tz_id': 'America/Bogota', 'localtime_epoch': 1725205217, 'localtime': '2024-09-01 10:40'}, 'current': {'last_updated_epoch': 1725204600, 'last_updated': '2024-09-01 10:30', 'temp_c': 23.2, 'temp_f': 73.8, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 10.5, 'wind_kph': 16.9, 'wind_degree': 150, 'wind_dir': 'SSE', 'pressure_mb': 1018.0, 'pressure_in': 30.06, 'precip_mm': 0.54, 'precip_in': 0.02, 'humidity': 73, 'cloud': 25, 'feelslike_c': 25.2, 'feelslike_f': 77.4, 'windchill_c': 22.2, 'windchill_f': 72.0, 'heatindex_c': 24.6, 'heatindex_f': 76.3, 'dewpoint_c': 18.8, 'dewpoint_f': 65.8, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 5.0, 'gust_mph': 15.0, 'gust_kph': 24.1}}", 'score': 0.9973042, 'raw_content': None}, {'title': 'Weather in January 2024 in Orlando, Florida, USA - timeanddate.com', 'url': 'https://www.timeanddate.com/weather/usa/orlando/historic?month=1&year=2024', 'content': 'Weather reports from January 2024 in Orlando, Florida, USA with highs and lows. Sign in. News. News Home; Astronomy News; ... See more current weather. ... January 2024 Weather in Orlando — Graph °F. See Hour-by-hour Forecast for upcoming weather.', 'score': 0.99343574, 'raw_content': None}], 'response_time': 2.77}
--
The| current| weather| in| Florida| is| partly| cloudy| with| a| temperature| of| |23|.|2|°C| (|73|.|8|°F|).| The| wind| is| coming| from| the| SSE| at| about| |16|.|9| k|ph| (|10|.|5| mph|).| The| humidity| level| is| at| |73|%| and| there| is| a| UV| index| of| |5|.|0|.|
"""
