import os
import json
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain.document_loaders import TextLoader
loader = TextLoader("TerryTurnerhighquality.txt")

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

answers = []
questions = [
    "Write 3 bullets with the highlights of the conversation for prospective business leaders",
    "When should managers think about recruiting?",
    "Why should managers be held accountable for retention targets?",
    "Why should exit interviews be interpreted through a lens of dissatisfaction"
]
with open("terry_summary_v2.txt", "w") as f:
    for q in questions:
        result = index.query_with_sources(q)
        f.write(result["question"]+"\n")
        f.write(result["answer"])

with open("terry_summary_v3")




