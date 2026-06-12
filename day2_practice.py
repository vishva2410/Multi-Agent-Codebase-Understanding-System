# day2_practice.py
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software engineer. Answer clearly and concisely."),
    ("human", "Explain what {topic} is in 3 bullet points for a junior developer.")
])

parser = StrOutputParser()
chain = prompt | llm | parser

if __name__ == "__main__":
    result = chain.invoke({"topic": "RAG (Retrieval Augmented Generation)"})
    print("=== Result 1 ===")
    print(result)

    result2 = chain.invoke({"topic": "vector embeddings"})
    print("\n=== Result 2 ===")
    print(result2)
