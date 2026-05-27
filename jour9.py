from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant expert en Python. Tu réponds en français, de manière claire et concise."),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

questions = [
    "C'est quoi une liste en Python ?",
    "Quelle est la différence entre un tuple et une liste ?",
    "C'est quoi un décorateur ?"
]

print("--- Assistant Python ---\n")
for question in questions:
    print(f"Question : {question}")
    reponse = chain.invoke({"question": question})
    print(f"Réponse : {reponse}\n")
    print("-" * 50)