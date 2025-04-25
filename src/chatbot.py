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
# Final function to process user query with handling of unrelated questions
def process_query(user_query):
    docs = get_top_documents_with_threshold(user_query)
    
    if not docs:
        return "Sorry, I couldn't find any related information in our system."

    response = qa_chain({
        "query": user_query,
        "context": docs  # override the context with filtered documents
    })

    return response["answer"]






# Create the conversational retrieval chain (with fallback context override)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    chain_type_kwargs={"prompt": get_prompt_template()}
)
