import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# Load the dataset
data = pd.read_csv("/content/qna_dataset.csv")  # Replace with your actual file path

# Combine Question and Answer into a single text
data["Combined"] = data.apply(
    lambda row: f"question: {row['question']} answer: {row['answer']}", axis=1
)
vector_db = Chroma(
        texts=combined_texts,
        metadatas=metadata,
        embedding_function=embeddings,
        persist_directory="db"  # Directory to persist the database
    )
    vector_db.persist()  # Save the vector database for later use
    return vector_db
embeddings = OpenAIEmbeddings()
# Extract combined texts and metadata
combined_texts = data["Combined"].tolist()
metadata = [{"question": row["question"], "answer": row["answer"]} for _, row in data.iterrows()]

# Print first few results for verification
print(combined_texts[:5])
print(metadata[:5])
