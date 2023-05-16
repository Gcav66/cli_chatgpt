import os
import sys
from dotenv import load_dotenv

import streamlit as st # import the Streamlit library
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


if str(sys.argv[1]) == "cro":
    from chat_prompts.cro import ST_TITLE, PLACEHOLDER_Q
if str(sys.argv[1]) == "bhf":
    from chat_prompts.bhf import ST_TITLE, PLACEHOLDER_Q
if str(sys.argv[1]) == "ciena":
    from chat_prompts.ciena import ST_TITLE, PLACEHOLDER_Q
if str(sys.argv[1]) == "transpo":
    from chat_prompts.transpo import ST_TITLE, PLACEHOLDER_Q
if str(sys.argv[1]) == "alb":
    from chat_prompts.alb import ST_TITLE, PLACEHOLDER_Q
if str(sys.argv[1]) == "spar":
    from chat_prompts.spar import ST_TITLE, PLACEHOLDER_Q


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set the title of the Streamlit app

#st.title("💰 Dataiku Document Search: LLM On Your Data")
st.title("💰 Dataiku Document Search: LLM On " + ST_TITLE +" Data")

# Add a link to the Github repository that inspired this app
st.markdown("Learn more about [Dataiku](https://www.dataiku.com/)")

# If an API key has been provided, create an OpenAI language model instance
if OPENAI_API_KEY:
    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
else:
    # If an API key hasn't been provided, display a warning message
    st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

# Add a text input box for the user's question
user_question = st.text_input(
    "Enter Your Question : ",
    #placeholder = "What are Cell and Gene Therapies?",
    placeholder = PLACEHOLDER_Q,
)

# Generating the final answer to the user's question using all the chains
if st.button("Tell me about it", type="primary"):
    # Chain 1: Generating a rephrased version of the user's question
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    embedding = OpenAIEmbeddings()
    persist_directory = 'db'
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
                                               vectorstore.as_retriever(), 
                                               memory=memory,
                                               #return_source_documents=True
                                               )

    #st.success(overall_chain.run(user_question)
    st.success(qa.run(user_question))
