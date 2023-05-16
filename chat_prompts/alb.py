BASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about Albemarle Corporation, a global manufacturer of specialty chemicals.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

MY_TEMPLATE = """You are an AI assistant for answering questions about Albemarle Corporation. 
You are given the following documents and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

SAMPLE_QUESTION = "What are Albemarle's core segments?"

ST_TITLE = "ALB Project AI - AI Chatbot"

PLACEHOLDER_Q = "What are Albemarle's core segments"