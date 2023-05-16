"""Python file to serve as the frontend"""
import os
from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_chain():
    embedding = OpenAIEmbeddings()
    persist_directory = 'db'
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
                                               vectorstore.as_retriever(), 
                                               memory=memory,
                                               #return_source_documents=True
                                               )
    return qa

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

chat_history = []
user_input = get_text()

if user_input:
    #output = chain.run(input=user_input)
    output = chain({"question": user_input, "chat_history": chat_history})

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")