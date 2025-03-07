import pandas as pd

# Load the dataset
data = pd.read_csv("/content/qna_dataset.csv")  # Replace with your actual file path

# Combine Question and Answer into a single text
data["Combined"] = data.apply(
    lambda row: f"question: {row['question']} answer: {row['answer']}", axis=1
)

# Extract combined texts and metadata
combined_texts = data["Combined"].tolist()
metadata = [{"question": row["question"], "answer": row["answer"]} for _, row in data.iterrows()]

# Print first few results for verification
print(combined_texts[:5])
print(metadata[:5])
