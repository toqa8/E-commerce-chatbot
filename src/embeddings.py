!pip install langchain-community
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# Load the dataset
data = pd.read_csv("/content/qna_dataset.csv")

# Combine Question and Answer into a single text
data["Combined"] = data.apply(
    lambda row: f"question: {row['question']} answer: {row['answer']}", axis=1
)

# Load Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2') # or other models like all-MiniLM-L6-v2

# Create documents for Chroma
documents = [
    Document(page_content=row["Combined"], metadata={"question": row["question"], "answer": row["answer"]})
    for _, row in data.iterrows()
]

# Create embeddings using Sentence Transformer
embeddings = model.encode([doc.page_content for doc in documents])

# Create the vector database using from_documents
vector_db = Chroma.from_texts(
    texts=[doc.page_content for doc in documents], # chroma from texts requires plain texts.
    embedding_function=model.encode, # Pass the encode function directly.
    metadatas=[doc.metadata for doc in documents], # pass the metadata.
    persist_directory="db"  # Directory to persist the database
)

vector_db.persist()  # Save the vector database for later use

# Print first few results for verification
print(embeddings[:5]) # print the embeddings.
