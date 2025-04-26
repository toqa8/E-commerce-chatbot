import streamlit as st
from chatbot import process_query
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Load vector DB
vector_db = Chroma(
    persist_directory="C:/Users/hp/db",
    embedding_function=embedding_function
)

st.title("E-commerce Assistant Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    response = process_query(user_input)
    st.write("Bot:", response)
