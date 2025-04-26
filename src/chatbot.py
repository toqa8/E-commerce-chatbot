from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Load vector DB
vector_db = Chroma(
    persist_directory="C:/Users/hp/db",
    embedding_function=embedding_function
)

# Load LLM
llm = OllamaLLM(model="llama2")

# Memory for tracking conversation history
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Prompt template
def get_prompt_template():
    template = """
You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

{context}

User's Question: {query}

Provide a clear and helpful answer based on the information above.
"""
    return PromptTemplate(input_variables=["context", "query"], template=template)

# Combine LLM with prompt
qa_prompt = get_prompt_template()
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# Create the retrieval chain
retriever = vector_db.as_retriever()
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Filter documents based on similarity threshold
def get_top_docs(query, k=5, threshold=1.2):
    results = vector_db.similarity_search_with_score(query, k=k)
    
    print("\nSimilarity Scores:")
    for doc, score in results:
        print(f"Score: {score:.4f} - {'Included' if score <= threshold else 'Excluded'}")

    filtered_docs = [doc for doc, score in results if score <= threshold]
    return filtered_docs

# Final function to process user query
def process_query(user_query):
    docs = get_top_docs(user_query)

    print(f"Filtered docs: {docs}")  # Debug

    if not docs:
        return "Sorry, I couldn't find any related information in our system."

    response = combine_docs_chain.invoke({
        "context": docs,
        "query": user_query
    })

    return response