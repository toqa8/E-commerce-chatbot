from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Load vector DB
vector_db = Chroma(
    persist_directory="D:/toka/depi/project/E-commerce-chatbot/src/db",
    embedding_function=embedding_function
)

# Print the number of embeddings (debugging)
print(f"Loaded {len(vector_db.get()["documents"])} embeddings.")

# Load LLM
llm = OllamaLLM(model="llama2", temperature=0)

# Prompt template
def get_prompt_template():
    template = """
You are a helpful e-commerce assistant. Use the following context (question-answer pairs) to help answer the user's question.

Context:
{context}

User's Question:
{question}

Provide a clear and helpful answer based on the information above.
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

# Get QA Prompt
qa_prompt = get_prompt_template()

# Wrap the LLM and prompt in an LLMChain (for question generation, not used yet directly)
llm_chain = LLMChain(llm=llm, prompt=qa_prompt)

# Conversation memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory
)

# Final function to process user query
def process_query(user_query):
    response = qa_chain.invoke({"question": user_query})
    real_response = response["answer"]
    return real_response
