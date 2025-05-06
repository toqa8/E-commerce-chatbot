from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
response = llm.invoke("What's the capital of Egypt?")
print("Bot:", response)
