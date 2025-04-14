from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """
You are a helpful e-commerce assistant. Use the following question-answer pairs (context) to help answer the user's question.

{context}

User's Question: {query}

Provide a clear and helpful answer based on the information above.
"""
    return PromptTemplate(input_variables=["context", "query"], template=template)
