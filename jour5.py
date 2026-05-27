from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

historique = []

print("chatbot pret! tapez 'exit' pour quitter")
print("-" * 40)

while True:
    message_utilisateur = input("Vous: ")

    if message_utilisateur.lower() == "exit":
        print("Au revoir!")
        break

    historique.append(
        {
            "role": "user",
            "content": message_utilisateur
        }
    )
    
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=historique
    )

    message_assistant = reponse.choices[0].message.content

    historique.append(
        {
            "role": "assistant",
            "content": message_assistant
        }
    )

    print(f"Assistant: {message_assistant}")
    print("-" * 40)