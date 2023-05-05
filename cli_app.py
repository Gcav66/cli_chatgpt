import pickle
import os
from dotenv import load_dotenv
from query_data import get_chain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#print (OPENAI_API_KEY)

if __name__ == "__main__":
    #with open("vectorstore.pkl", "rb") as f:
    #    vectorstore = pickle.load(f)
    embedding = OpenAIEmbeddings()
    persist_directory = 'db'
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
                                               vectorstore.as_retriever(), 
                                               memory=memory,
                                               #return_source_documents=True
                                               )
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
