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

st.set_page_config(page_title="E-commerce Assistant", page_icon="🛒")
st.title("🛒 E-commerce Assistant Chatbot")

st.markdown(
    """
    <style>
    .stChatMessage {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything about products or orders!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    response = process_query(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

# streamlit run d:\toka\depi\project\E-commerce-chatbot\src\streamlit_app.py
