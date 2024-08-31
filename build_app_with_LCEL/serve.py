import os

import dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

system_template = "Translate the following info {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}'),
])
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
