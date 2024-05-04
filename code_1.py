from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import os

loader = DirectoryLoader("C:\\Users\\A\\Desktop\\All Subjects\\Gen AI\\Books Data")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)

texts = text_splitter.split_documents(documents)

print(texts[0])

openai_api_key = "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"
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

vector_db = FAISS.from_documents(texts, embeddings)

llm = OpenAI(openai_api_key = openai_api_key)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",\
                                 retriever = vector_db.as_retrieval()
                                )

query = "What is machine learning"
qa.run(query)