# agents/base.py
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_llm(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Free tier, fast
        temperature=temperature
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

def get_parser():
    return StrOutputParser()
