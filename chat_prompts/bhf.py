BASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about annuity financial products.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

MY_TEMPLATE = """You are an AI assistant for answering questions about financial annuity products, specifically fixed, variable, and index-linked annuities. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

ST_TITLE = "Brighthouse SHIELD Annuity"

PLACEHOLDER_Q = "If I have a Shield Rate of 15%, And the market declines 20%, how does that affect my account value?"

SAMPLE_QUESTION = "What is a index annuity?"