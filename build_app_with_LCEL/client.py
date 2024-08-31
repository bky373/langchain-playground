from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
result = remote_chain.invoke({"language": "Korean", "text": "hi"})
print(result)  # 안녕하세요!
