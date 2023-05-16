BASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about products sold by Electrical Wholesalers.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

MY_TEMPLATE = """You are an AI assistant for answering questions about the Electrical Wholesaling Industry. 
You are given the following documents and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

SAMPLE_QUESTION = "What are the biggest drivers of sales in 2022?"

ST_TITLE = "Electrical Economy - AI Chatbot"

PLACEHOLDER_Q = "What do Distributors expect for 2023 growth?"