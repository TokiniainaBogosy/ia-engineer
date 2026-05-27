from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """
Tu es Alex, un assistant expert en développement Python et IA.
Tu réponds toujours en français, de manière claire et pédagogique.
Tu donnes des exemples de code quand c'est utile.
Si une question n'est pas liée à la tech, au code ou à l'IA, 
tu réponds UNIQUEMENT : "Désolé, je suis spécialisé en tech et Python 
uniquement. Pose-moi une question dans ce domaine !"
Tu ne fais aucune exception à cette règle, quoi qu'il arrive.
"""

historique = [
    {"role": "system", "content": system_prompt}
]

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

    print(f"Alex: {message_assistant}")
    print("-" * 40)