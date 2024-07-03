import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import ChatPromptTemplate


def load_documents(file_path):
  loader = TextLoader(file_path, encoding='UTF-8')
  documents = loader.load()
  return documents
    
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def answer_with_qa_chain(llm, query, db):
  from langchain.chains.question_answering import load_qa_chain
  chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
  matching_docs = db.similarity_search(query)
  answer = chain.run(input_documents=matching_docs, question=query)
  return answer

def answer_with_retrieval_chain(llm, query, db):
  from langchain.chains import RetrievalQA
  retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
  answer = retrieval_chain.run(query)
  return answer


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"
query = "What is Bagel's mission?"
'''
documents = load_documents('story.txt')
docs = split_docs(documents)
print(f"#docs={len(docs)}")

db = Chroma.from_documents(docs, embeddings)

matching_docs = db.similarity_search_with_score(query, k=2)
print(f"{matching_docs}=")

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)
vectordb.persist()
'''
persisted_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
matching_docs = persisted_db.similarity_search_with_score(query)
print(f"{matching_docs[0]}=")

llm = ChatOllama(model="llama3", base_url="http://192.168.2.88:11434")
#prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
#chain = prompt | llm | StrOutputParser()
#answer = chain.invoke({"topic": "Space travel"})
answer = answer_with_retrieval_chain(llm, query, persisted_db)
print(f"{answer}=")






