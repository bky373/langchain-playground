import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

dotenv.load_dotenv()
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
# batch_result = retriever.batch(["cat", "shark"])
# print(batch_result)
"""
[[Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.')], 
[Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]]
"""

as_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
# as_retriever_batch = as_retriever.batch(["cat", "shark"])
# print(as_retriever_batch)
"""
[[Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.')], 
[Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]]
"""

llm = ChatOpenAI(model="gpt-4o-mini")
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])
rag_chain = {"context": as_retriever, "question": RunnablePassthrough()} | prompt | llm
rag_chain_invoke = rag_chain.invoke("tell me about cats")
print(rag_chain_invoke)
"""content='Cats are independent pets that often enjoy their own space.' additional_kwargs={'refusal': None} 
response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 53, 'total_tokens': 64}, 'model_name': 
'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f33667828e', 'finish_reason': 'stop', 'logprobs': None} 
id='run-9764c81c-79c5-4318-bef4-a6e3755f4b03-0' usage_metadata={'input_tokens': 53, 'output_tokens': 11, 
'total_tokens': 64}"""
