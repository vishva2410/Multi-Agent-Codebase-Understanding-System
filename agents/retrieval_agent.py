# agents/retrieval_agent.py
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

load_dotenv()

class RetrievalAgent:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.parser = StrOutputParser()

    def index_codebase(self, code_chunks: List[Dict[str, Any]]):
        print(f"Indexing {len(code_chunks)} code chunks...")
        documents = []
        for chunk in code_chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "source": chunk.get("source", "unknown"),
                    "module": chunk.get("module", "unknown"),
                    "language": chunk.get("language", "python")
                }
            )
            documents.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\ndef ", "\nclass ", "\n\n", "\n"]
        )
        split_docs = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"Indexed {len(split_docs)} chunks into ChromaDB")

    def load_existing_index(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        print(f"Loaded index with {self.vectorstore._collection.count()} chunks")

    def answer_question(self, question: str, module_filter: str = None) -> Dict:
        if not self.vectorstore:
            raise Exception("No index loaded.")

        search_kwargs = {"k": 4}
        if module_filter:
            search_kwargs["filter"] = {"module": module_filter}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(question)

        context_parts = []
        sources = set()
        for doc in relevant_docs:
            source = doc.metadata.get("source", "unknown")
            sources.add(source)
            context_parts.append(f"# File: {source}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code analyst. Answer questions about the codebase
            using ONLY the provided code context. Be specific and reference function names,
            file names, and line logic. If the answer isn't in the context, clearly say so.

            Codebase context:
            {context}
            """),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": list(sources),
            "chunks_used": len(relevant_docs)
        }


if __name__ == "__main__":
    agent = RetrievalAgent()
    agent.load_existing_index()

    questions = [
        "How does authentication work?",
        "Where is payment processing implemented?",
        "How is a JWT token validated?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = agent.answer_question(q)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("="*50)
