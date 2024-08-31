import os

import dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

model = ChatOpenAI(model="gpt-4o-mini")
# result = model.invoke([HumanMessage(content="Hi! I'm Borahm")])
# print(result)
# AIMessage(content='Hi Borahm! How can I assist you today?' additional_kwargs={'refusal': None} response_metadata={
# 'token_usage': {'completion_tokens': 12, 'prompt_tokens': 13, 'total_tokens': 25}, 'model_name':
# 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'stop', 'logprobs': None}
# id='run-e80df47c-ff49-4e2a-a9d6-6837fc95906c-0' usage_metadata={'input_tokens': 13, 'output_tokens': 12,
# 'total_tokens': 25})
# https://smith.langchain.com/public/f0f71198-db1d-4579-bb11-2ac0498d63ad/r

# next_result = model.invoke([HumanMessage(content="What's my name?")])
# print(next_result)
# AIMessage(content="I don't know your name. If you'd like to share it or have me refer to you by a certain name,
# feel free to let me know!" additional_kwargs={'refusal': None} response_metadata={'token_usage': {
# 'completion_tokens': 30, 'prompt_tokens': 11, 'total_tokens': 41}, 'model_name': 'gpt-4o-mini-2024-07-18',
# 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'stop', 'logprobs': None}
# id='run-12c51967-cfc1-46f8-8709-c5f83d5063d4-0' usage_metadata={'input_tokens': 11, 'output_tokens': 30,
# 'total_tokens': 41})
# https://smith.langchain.com/public/9e0c377e-11d8-4a72-8ee1-59354bffe762/r
result2 = model.invoke(
    [
        HumanMessage(content="Hi! I'm Borahm"),
        AIMessage(content="Hi Borahm! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
print(result2)
# AIMessage(content='Your name is Borahm. How can I help you today?' additional_kwargs={'refusal': None}
# response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 37, 'total_tokens': 51}, 'model_name':
# 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'stop', 'logprobs': None}
# id='run-d7154fb2-d4bf-4df3-8c3c-79e3c77a497d-0' usage_metadata={'input_tokens': 37, 'output_tokens': 14,
# 'total_tokens': 51})
# https://smith.langchain.com/public/2fe43acd-3fe1-4381-82c0-ed35ce0cde52/r
