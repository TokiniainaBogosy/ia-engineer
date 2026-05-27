from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu es un assistant sympa qui répond en français. Tu te souviens de toute la conversation."),
        MessagesPlaceholder(variable_name="historique"),
        ("human", "{question}")
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

historique = []

print("Chatbot LangChain (tape 'exit' pour quitter)")
print("-" * 50)

while True:
    question = input("Vous : ")
    if question.lower() == "exit":
        print("Au revoir !")
        break
    
    # Garder uniquement les 10 derniers messages
    if len(historique) > 10:
        historique = historique[-10:]

    reponse = chain.invoke({
        "historique": historique,
        "question": question
    })

    historique.append(HumanMessage(content=question))
    historique.append(AIMessage(content=reponse))

    print(f"Chatbot : {reponse}\n")
    print("-" * 50)
