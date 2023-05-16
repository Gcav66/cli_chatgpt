"""Python file to serve as the frontend"""
import sys
import os
import pickle
from dotenv import load_dotenv
import openai
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

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
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.document_loaders import YoutubeLoader


if str(sys.argv[2]) == "cro":
    from chat_prompts.cro import SAMPLE_QUESTION
if str(sys.argv[2]) == "bhf":
    from chat_prompts.bhf import SAMPLE_QUESTION
if str(sys.argv[2]) == "ciena":
    from chat_prompts.ciena import SAMPLE_QUESTION 
if str(sys.argv[2]) == "alb":
    from chat_prompts.alb import SAMPLE_QUESTION
if str(sys.argv[2]) == "transpo":
    from chat_prompts.transpo import SAMPLE_QUESTION 
if str(sys.argv[2]) == "spar":
    from chat_prompts.spar import SAMPLE_QUESTION

logging.info(SAMPLE_QUESTION)

if sys.argv[3] == "yt":
    yt_url = sys.argv[1]
    yt_loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=False)
    pages = yt_loader.load_and_split()

if sys.argv[3] == "pdf":
    pdf_path = str(sys.argv[1])
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()


print(pages[0].page_content)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory="db")
vectordb.persist()

pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                    vectordb, return_source_documents=True)

#query = "What are Cell and Gene Therapies?"
#query = "What is a index annuity?"
query = SAMPLE_QUESTION
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])