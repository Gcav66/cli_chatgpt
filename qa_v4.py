from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.chains import VectorDBQAWithSourcesChain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Example of your response should be:

```
The answer is foo
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.langchain_factory
def init():
    #file = None
    file = "/Users/guscavanaugh/Documents/accounts_docs/avalonbay/Home-Maintenance-Guide.pdf"

    # Wait for the user to upload a file
    while file == None:
        file = cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    # Decode the file
    #text = file.content.decode("utf-8")

    pdf_path = "/Users/guscavanaugh/Documents/accounts_docs/avalonbay/Home-Maintenance-Guide.pdf"
    pdf_loader = PyPDFLoader(pdf_path)
    texts = pdf_loader.load_and_split()
    
    # Split the text into chunks
    #texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    #docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore = Chroma.from_documents(texts, embeddings, metadatas=metadatas)


    # Create a chain that uses the Chroma vector store
    #chain = VectorDBQAWithSourcesChain.from_chain_type(
    #    ChatOpenAI(temperature=0, max_tokens=1000),
    #    vectorstore=vectorstore,
    #    return_source_documents=True,
    #    chain_type_kwargs=chain_type_kwargs,
    #    reduce_k_below_max_tokens=True
    #)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, max_tokens=1000),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    #cl.Message(content=f"`{file.name}` uploaded, you can now ask questions!").send()
    cl.Message(content="Document uploaded, you can now ask questions!").send()

    return chain



@cl.langchain_postprocess
def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(text=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            #answer += "\nNo sources found"
            answer += f"\nSources: /Users/guscavanaugh/Documents/accounts_docs/avalonbay/Home-Maintenance-Guide.pdf"
            #answer += f"\nSources: {', '.join(pdf_path)}"

    cl.Message(content=answer, elements=source_elements).send()
