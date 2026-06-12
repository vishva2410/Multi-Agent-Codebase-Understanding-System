# day3_rag.py
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Step 1: Loading document...")
loader = TextLoader("data/sample_code.py")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

print("\nStep 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\ndef ", "\nclass ", "\n\n", "\n", " "]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("\nStep 3: Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Embeddings stored!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a code assistant. Answer questions about the codebase
    using ONLY the provided context. If the answer isn't in the context, say so.

    Context from codebase:
    {context}
    """),
    ("human", "{question}")
])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | parser
)

print("\n" + "="*50)
questions = [
    "How does authentication work?",
    "How is a JWT token created?",
    "How does password hashing work?",
    "Is there any payment processing code?"
]

for q in questions:
    print(f"Q: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 40)
