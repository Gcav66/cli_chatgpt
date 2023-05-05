from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ChatVectorDBChain
import openai
import os
from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains import ConversationalRetrievalChain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about scientific advice for health and wellness.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about bodybuilding. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    #pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    #                                vectordb, return_source_documents=True)
    #llm = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain