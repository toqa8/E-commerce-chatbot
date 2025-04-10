import pandas as pd
from langchain_community.vectorstores import Chroma # a vectore store to store embeddings and metadata
from langchain.docstore.document import Document # a standard format for storing text and metadata
from langchain_huggingface import HuggingFaceEmbeddings # to load a hugging face sentence transformer model
import os

# Load the dataset
data = pd.read_csv(r"D:\toka\depi\project\E-commerce-chatbot\data\qna_dataset.csv")

# Creates a new column "Combined" Combining Question and Answer into a single text
data["Combined"] = data.apply(
    lambda row: f"question: {row['question']} answer: {row['answer']}", axis=1
)

# Create documents for Chroma, instead of plain text from the combined text, for more flexibility for more use cases and for working smoothly in langchain ecosystem
documents = [
    Document(page_content=row["Combined"], metadata={"question": row["question"], "answer": row["answer"]})
    for _, row in data.iterrows()
]

# Load the embedding model using LangChain's wrapper
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

try:
# Create the vector database using from_documents
    vector_db = Chroma.from_documents(
        documents,
        embedding=embedding_function,  # Correct: pass the wrapper, not just a function
        persist_directory="db"
    )
    print('Vector store created successfully')
except Exception as e:
    print(f"error: {e}")

print("Saving DB to:", os.path.abspath("db"))