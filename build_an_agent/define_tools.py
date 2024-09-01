import dotenv
from langchain_community.tools import TavilySearchResults

dotenv.load_dotenv()
search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
print(search_results)
"""[{'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'San Francisco', 'region': 'California', 
'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 
'localtime_epoch': 1725203202, 'localtime': '2024-09-01 08:06'}, 'current': {'last_updated_epoch': 1725202800, 
'last_updated': '2024-09-01 08:00', 'temp_c': 16.7, 'temp_f': 62.1, 'is_day': 1, 'condition': {'text': 'Partly 
cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 9.4, 'wind_kph': 15.1, 
'wind_degree': 250, 'wind_dir': 'WSW', 'pressure_mb': 1015.0, 'pressure_in': 29.97, 'precip_mm': 0.0, 'precip_in': 
0.0, 'humidity': 86, 'cloud': 50, 'feelslike_c': 16.7, 'feelslike_f': 62.1, 'windchill_c': 14.0, 'windchill_f': 57.3, 
'heatindex_c': 14.4, 'heatindex_f': 58.0, 'dewpoint_c': 12.3, 'dewpoint_f': 54.1, 'vis_km': 16.0, 'vis_miles': 9.0, 
'uv': 4.0, 'gust_mph': 11.0, 'gust_kph': 17.6}}"}, {'url': 
'https://www.meteoprog.com/weather/Sanfrancisco/month/january/', 'content': 'San Francisco (United States) weather in 
January 2024 ☀️ Accurate weather forecast for San Francisco in January ⛅ Detailed forecast By month Current 
temperature "near me" Weather news ⊳ Widget of weather ⊳ Water temperature | METEOPROG.COM'}]"""
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]
print(tools)  # [TavilySearchResults(max_results=2)]
