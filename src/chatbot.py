from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model='llama2')


def get_prompt_template():
    template = """
You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

{context}

User's Question: {query}

Provide a clear and helpful answer based on the information above.
"""
    return PromptTemplate(input_variables=["context", "query"], template=template)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory

# Add memory to keep track of conversation history
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
