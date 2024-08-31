import os

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")
messages = [
    SystemMessage(content="Translate the following from English into Korean"),
    HumanMessage(content="Hi!"),
]

# result = model.invoke(messages)
# print(result)
# AIMessage(content='안녕하세요!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'stop', 'logprobs': None} id='run-cf809816-f888-4500-a4fa-097493254214-0' usage_metadata={'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23})


# parser = StrOutputParser()
# parsed_result = parser.invoke(result)
# print(parsed_result)  # 안녕하세요!
# chain = model | parser
# chain_result = chain.invoke(messages)
# print(chain_result)  # 안녕하세요!
# LangSmith trace: https://smith.langchain.com/public/eea6c107-f628-4707-95fe-168b521ebf11/r

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt_template_result = prompt_template.invoke({"language": "Korean", "text": "hi"})

# ChatPromptValue(messages=[SystemMessage(content='Translate the following into italian:'), HumanMessage(content='hi')])
print(prompt_template_result)
to_messages = prompt_template_result.to_messages()
print(to_messages)  # [SystemMessage(content='Translate the following into Korean:'), HumanMessage(content='hi')]
