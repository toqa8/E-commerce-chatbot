from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

import sys
print(sys.executable)

english_df = pd.read_csv(r"D:\toka\depi\project\E-commerce-chatbot\data\qna_dataset.csv")

# Load English-to-Arabic translation model
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Translate question and answer columns
english_df["question_ar"] = english_df["question"].apply(translate)
english_df["answer_ar"] = english_df["answer"].apply(translate)

arabic_df = english_df[["question_ar", "answer_ar"]]
arabic_df.columns = ["question", "answer"]  # Rename for consistency
arabic_df.to_csv(r"D:\toka\depi\project\E-commerce-chatbot\data\qna_dataset_ar.csv", index=False)