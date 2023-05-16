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
from chat_prompts.cro import BASE_TEMPLATE, MY_TEMPLATE

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_template = BASE_TEMPLATE
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = MY_TEMPLATE
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