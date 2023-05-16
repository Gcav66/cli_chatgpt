BASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about freight and parcel shipping.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

MY_TEMPLATE = """You are an AI assistant for answering questions about freight and parcel shipping. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

SAMPLE_QUESTION = "What are the core macroeconomic trends facing Shippers?"

ST_TITLE = "TI.AI - Shipper Insights"

PLACEHOLDER_Q = "What are the core macroeconomic trends facing Shippers?"