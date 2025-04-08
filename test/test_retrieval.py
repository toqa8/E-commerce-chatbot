from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load the same embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Reload the existing DB (no re-encoding or re-saving)
vector_db = Chroma(
    persist_directory="db",
    embedding_function=embedding_function
)

query = "how can i return a product?"
results = vector_db.similarity_search(query, k=2)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print("Content:", result.page_content)
    print("Metadata:", result.metadata)