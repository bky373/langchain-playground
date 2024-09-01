import datetime

import dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

# Invocation: Invoke directly with args
# result = tool.invoke({"query": "What happened at the last wimbledon"})
# print(result)
"""[{'url': 'https://www.theguardian.com/sport/live/2023/jul/16/wimbledon-mens-singles-final-2023-carlos-alcaraz-v
-novak-djokovic-live?page=with:block-64b3ff568f08df28470056bf', 'content': 'Carlos Alcaraz recovered from a set down 
to topple Djokovic 1-6, 7-6(6), 6-1, 3-6, 6-4 and win his first Wimbledon title in a battle for the ages'}, 
{'url': 'https://www.cnn.com/2022/07/10/tennis/wimbledon-2022-mens-final-djokovic-kyrgios-spt-intl/index.html', 
'content': '“It is the most special tennis court in the world and when you walk on the untouched grass and everything 
is so directed on the tennis, the players’ ball and racquet and it has the most recognition in the world.”\n In the 
next game, the pressure began to look like it was telling on Kyrgios as Djokovic broke the 27-year-old, resulting in 
a berating in the direction of his box in the break between games.\n Kyrgios unique style of play, including his 
occasional chuntering to his box, has won over the watching crowds and it was the same for the final, 
with the 27-year-old receiving much of the backing.\n And in the fourth game of the set, he broke Kyrgios to take a 
two-game lead – the first time he’s broke the Australian in their three meetings.\n Djokovic, playing in his 32nd 
grand slam final, has often been the less-favored of the players when he’s played at Wimbledon, often coming up 
against Nadal and Roger Federer, and it was the same during Sunday’s final.\n'}, 
{'url': 'https://www.cnn.com/2024/07/09/sport/novak-djokovic-wimbledon-crowd-quarterfinals-spt-intl/index.html', 
'content': 'Novak Djokovic produced another impressive performance at Wimbledon on Monday to cruise into the 
quarterfinals, but the 24-time grand slam champion was far from happy after his win. The Serb took ...'}, 
{'url': 'https://www.cnn.com/2024/07/05/sport/andy-murray-wimbledon-farewell-ceremony-spt-intl/index.html', 
'content': "It was an emotional night for three-time grand slam champion Andy Murray on Thursday, 
as the 37-year-old's Wimbledon farewell began with doubles defeat.. Murray will retire from the sport this ..."}, 
{'url': 'https://www.bbc.com/sport/tennis/articles/cw0yqgvyx19o', 'content': 'Elsewhere, Croatian Donna Vekic 
progressed to the last eight by overcoming Spaniard Paula Badosa in three sets. The world number 37 won 6-2 1-6 6-4 
in a rain-interrupted game on court two.'}]
"""

# Invocation: Invoke with ToolCall
# model_generated_tool_call = {
#     "args": {"query": "euro 2024 host nation"},
#     "id": 1,
#     "name": "tavily",
#     "type": "tool_call",
# }
# tool_msg = tool.invoke(model_generated_tool_call)

# The content is a JSON string of results
# print(tool_msg.content[:400])
"""[{"url": "https://www.sportingnews.com/uk/football/news/list-euros-host-nations-uefa-european-championship
-countries/85f8069d69c9f4ecd00c4900", "content": "The 2024 UEFA European Championship, more commonly known as Euro 
2024, will mark the 17th edition of the tournament and is set to be hosted by Germany from June 14 to July 14."}, 
{"url": "https://www.sportingnews.com/us/soccer/news/where-euro-2"""

# # The artifact is a dict with richer, raw results
# {k: type(v) for k, v in tool_msg.artifact.items()}
#
# {'query': str,
#  'follow_up_questions': NoneType,
#  'answer': str,
#  'images': list,
#  'results': list,
#  'response_time': float}

# Abbreviate the results for demo purposes
# print(json.dumps({k: str(v)[:200] for k, v in tool_msg.artifact.items()}, indent=2))
"""
{
  "query": "euro 2024 host nation",
  "follow_up_questions": "None",
  "answer": "Germany will be the host nation for Euro 2024, with the tournament scheduled to take place from June 14 to July 14. The matches will be held in 10 different cities across Germany, including Berlin, Co",
  "images": "['https://i.ytimg.com/vi/3hsX0vLatNw/maxresdefault.jpg', 'https://img.planetafobal.com/2021/10/sedes-uefa-euro-2024-alemania-fg.jpg', 'https://editorial.uefa.com/resources/0274-14fe4fafd0d4-413fc8a7b7",
  "results": "[{'title': 'Where is Euro 2024? Country, host cities and venues', 'url': 'https://www.radiotimes.com/tv/sport/football/euro-2024-location/', 'content': \"Euro 2024 host cities. Germany have 10 host cit",
  "response_time": "3.97"
}
"""

# Chaining
llm = ChatOpenAI(model="gpt-4o-mini")

today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# specifying tool_choice will force the model to call this tool.
llm_with_tools = llm.bind_tools([tool])
llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    """
    첫 번째 llm_chain.invoke(input_, config=config):
    - 목적: 사용자의 초기 입력에 대한 AI의 첫 번째 응답을 생성합니다.
    - 설명: 이 단계에서 AI는 사용자의 질문을 이해하고, 필요한 경우 추가 정보를 얻기 위해 어떤 도구를 사용해야 할지 결정합니다.
    - 결과: 'ai_msg'로, AI의 초기 응답과 필요한 도구 호출 정보를 포함합니다.
    tool.batch(ai_msg.tool_calls, config=config):
    - 목적: AI가 요청한 도구들을 실행합니다.
    - 설명: 첫 번째 단계에서 AI가 결정한 도구들을 실제로 호출하여 필요한 정보를 가져옵니다.
    - 결과: 'tool_msgs'로, 각 도구 호출의 결과를 포함합니다.
    두 번째 llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config):
    - 목적: 최종 응답을 생성합니다.
    - 설명: 초기 AI 응답(ai_msg)과 도구 실행 결과(tool_msgs)를 모두 고려하여 더 정확하고 상세한 최종 응답을 생성합니다.
    - 결과: 사용자의 질문에 대한 최종 응답입니다.
    이러한 과정을 거치는 이유:
    - 복잡한 질문에 대해 더 정확하게 대응할 수 있습니다.
    - 실시간 또는 특정 정보가 필요한 경우, 도구를 통해 최신 정보를 얻을 수 있습니다.
    - AI의 초기 추론과 실제 데이터를 결합하여 더 신뢰할 수 있는 응답을 제공합니다.
    이 방식은 AI가 단순히 학습된 정보만을 제공하는 것이 아니라, 필요한 경우 외부 도구를 활용하여 더 정확하고 최신의 정보를 제공할 수 있게 해줍니다.
    """
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    print(f"ai_msg: {ai_msg}")
    # ai_msg: AIMessage(content='' additional_kwargs={'tool_calls': [{'id': 'call_Za9N5avNf58MItAnussZ6I2h',
    # 'function': { 'arguments': '{"query":"last women\'s singles Wimbledon winner 2024"}',
    # 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage':
    # {'completion_tokens': 25, 'prompt_tokens': 104, 'total_tokens': 129}, 'model_name': 'gpt-4o-mini-2024-07-18',
    # 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'tool_calls', 'logprobs': None})
    # id='run-29db9bb4-4ad5-49a5-bef0-f62873ad17f2-0' tool_calls=[{'name': 'tavily_search_results_json',
    # 'args': {'query': "last women's singles Wimbledon winner 2024"}, 'id': 'call_Za9N5avNf58MItAnussZ6I2h',
    # 'type': 'tool_call'}] usage_metadata={'input_tokens': 104, 'output_tokens': 25, 'total_tokens': 129}

    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    print(f"tool_msgs: {tool_msgs}")  # tool_msgs: [ToolMessage(content='[{"url":
    # "https://sports.ndtv.com/wimbledon-2024/wimbledon-2024-live-score-womens-singles-final-tennis-barbora
    # -krejcikova-vs-jasmine-paolini-live-updates-6097625", "content": "Wimbledon 2024 Women\'s Singles Final
    # Highlights: Barbora Krejcikova won her first Wimbledon crown, and her second Grand Slam title overall."},

    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


response = tool_chain.invoke("who won the last women's singles wimbledon")
print(response)
# AIMessage(content="The last women's singles Wimbledon champion is Barbora Krejcikova, who won her first Wimbledon
# title by defeating Jasmine Paolini with a score of 6-2, 2-6, 6-4 in the final. This victory, which took place at
# the 2024 Wimbledon Championships, marked her second Grand Slam title overall." additional_kwargs={'refusal': None}
# response_metadata={'token_usage': {'completion_tokens': 70, 'prompt_tokens': 675, 'total_tokens': 745},
# 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_9722793223', 'finish_reason': 'stop', 'logprobs':
# None} id='run-50eb894a-9c86-420d-b20b-563e996d5cdb-0' usage_metadata={'input_tokens': 675, 'output_tokens': 70,
# 'total_tokens': 745})
