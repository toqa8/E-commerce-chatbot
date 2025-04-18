from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

vector_db = Chroma(
    persist_directory="db",
    embedding_function=embedding_function
)

# Import your prompt template function
def get_prompt_template():
    template = """
You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

{context}

User's Question: {query}

Provide a clear and helpful answer based on the information above.
"""
    return PromptTemplate(input_variables=["context", "query"], template=template)

# Simulate user input
user_query = "Recommend hair products"

# Use the vector database to retrieve relevant docs
retrieved_docs = vector_db.similarity_search(user_query)

# Build the context from retrieved documents
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Get the prompt template and format it
prompt_template = get_prompt_template()
formatted_prompt = prompt_template.format(context=context, query=user_query)

# Print the formatted prompt
print(formatted_prompt)




# OUTPUT
# You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

# question: What is the rating of Styling Shampoo For Men - Cooling & Style? answer: The rating of Styling Shampoo For Men - Cooling & Style is 5.0
# question: What is the rating of Styling Shampoo For Men - Cooling & Style? answer: The rating of Styling Shampoo For Men - Cooling & Style is 5.0
# question: What is the rating of Supreme Scalp Rejuvenation Shampoo? answer: The rating of Supreme Scalp Rejuvenation Shampoo is 5.0
# question: What is the rating of Hair Care Kit - Oil, Shampoo, Conditioner & Serum? answer: The rating of Hair Care Kit - Oil, Shampoo, Conditioner & Serum is 4.1

# User's Question: Recommend hair products

# Provide a clear and helpful answer based on the information above.
