from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def calculer_facture(description):
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """Tu es un assistant comptable.
                Quand on te donne une description de facture, tu calcules le total.
                Raisonne étape par étape avant de donner la réponse finale, et montre ton calcul.
                Termine TOUJOURS par une ligne : TOTAL : X,XX€"""
            },
            {
                "role": "user",
                "content": description
            }
        ]
    )
    return reponse.choices[0].message.content.strip()

factures = [
    "3 articles à 24,99€, réduction de 20%, frais de port 5,90€",
    "2 abonnements à 9,99€/mois pendant 6 mois, TVA 20%",
    "Prestation de 450€, remise fidélité 15%, TVA 20%"
]

print("--- Calcul de factures ---\n")
for facture in factures:
    print(f"Facture : {facture}")
    print(calculer_facture(facture))
    print("-" * 50)