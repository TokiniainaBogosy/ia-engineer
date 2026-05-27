from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyser_sentiment(texte):
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Tu es un expert en analyse de sentiment. Réponds UNIQUEMENT par : positif, négatif, ou mitigé."
            },
            {
                "role": "user",
                "content": "Super produit, je recommande !"
            },
            {
                "role": "assistant",
                "content": "positif"
            },
            {
                "role": "user",
                "content": "Vraiment décevant, ne commandez pas."
            },
            {
                "role": "assistant",
                "content": "négatif"
            },
            {
                "role": "user",
                "content": "C'est pas mal, mais ça pourrait être mieux."
            },
            {
                "role": "assistant",
                "content": "mitigé"
            },
            {
                "role": "user",
                "content": texte
            }
        ]
    )
    return reponse.choices[0].message.content.strip()

textes = [
    "Livraison rapide mais emballage abîmé.",
    "Exactement ce que je voulais, parfait !",
    "Jamais reçu ma commande, service client inexistant.",
    "Pas mal pour le prix."
]

print("--- Analyse de sentiment ---")
for texte in textes:
    sentiment = analyser_sentiment(texte)
    print(f"{sentiment:10} | {texte}")