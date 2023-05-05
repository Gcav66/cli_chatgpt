"""Python file to serve as the frontend"""
import os
import pickle
from dotenv import load_dotenv
import openai


#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import UnstructuredFileLoader
#from langchain.vectorstores.faiss import FAISS
#from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Load Data
#loader = UnstructuredFileLoader("boiler.txt")
#loader = UnstructuredFileLoader("sample_txts/hany_butler_chest.txt")
#raw_documents = loader.load()

# Split text
#text_splitter = RecursiveCharacterTextSplitter()
#documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
#embeddings = OpenAIEmbeddings()
#my_vectorstore = FAISS.from_documents(documents, embeddings)

# Save vectorstore
#with open("vectorstore.pkl", "wb") as f:
#    pickle.dump(my_vectorstore, f)

from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)

pdf_path = "sample_txts/neat.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(pages[0].page_content)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory="db")
vectordb.persist()

pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                    vectordb, return_source_documents=True)

query = "What is NEAT"
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])