

"""**Test**"""

from langchain.prompts import PromptTemplate
import pandas as pd

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
user_query = "How long does delivery take?"

# Use the vector database to retrieve relevant docs
retrieved_docs = vector_db.similarity_search(user_query)

# Build the context from retrieved documents
context = "\n".join([doc.page_content for doc in retrieved_docs])


# Get the prompt template and format it
prompt_template = get_prompt_template()
formatted_prompt = prompt_template.format(context=context, query=user_query)

# Print the formatted prompt
print(formatted_prompt)


#OUTPUT
# You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

# question: What is the discount on Condoms - Extra Time? answer: The discount on Condoms - Extra Time is 0.0%
# question: What is the discount on For Boys - With Surprise? answer: The discount on For Boys - With Surprise is 0.0%
# question: What is the discount on Rusk - Baby? answer: The discount on Rusk - Baby is 0.0%
# question: What is the discount on Papad - Potato? answer: The discount on Papad - Potato is 0.0%

# User's Question: How long does delivery take?

# Provide a clear and helpful answer based on the information above
