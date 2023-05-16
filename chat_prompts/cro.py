BASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about clinical research for pharmaceutical companies.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

MY_TEMPLATE = """You are an AI assistant for answering questions about clinical trial research. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

SAMPLE_QUESTION = "What are Cell and Gene Therapies?"

ST_TITLE = "CRO Product Launch"

PLACEHOLDER_Q = "Which CGT use cases exceeded expectations?"