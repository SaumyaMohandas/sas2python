from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv


import streamlit as st

api_key = st.secrets["GOOGLE_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)


db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
print("✅ Embeddings saved to faiss_index/")
