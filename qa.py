import os
import sys
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

#pdf_path = str(sys.argv[1])
pdf_path = "/Users/guscavanaugh/Documents/accounts_docs/avalonbay/Home-Maintenance-Guide.pdf"
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory="db")
vectordb.persist()

#pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
#                                    vectordb, return_source_documents=True)

template = """
Use the following pieces of context to answer the users question
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

@cl.langchain_factory
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    #llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), verbose=True)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", max_tokens=2000),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

    #return llm_chain
    return chain
