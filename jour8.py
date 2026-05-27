from dotenv import load_dotenv
import os
from groq import Groq
from pydantic import BaseModel
import json
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Facture(BaseModel):
    numero : str
    date : str
    client : str 
    montant_ht : float
    tva : float
    montant_ttc : float

def extraire_facture(texte):
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """Tu es un extracteur de données de factures.
                Tu reponds UNIQUEMENT en format JSON valide, sans texte avant ou apres, sans balises markdown.
                Tu convertis TOUJOURS les dates au format ISO 8601 : YYYY-MM-DD.
                Format attendu:
                {
                    "numero": "...",
                    "date": "...",
                    "client": "...",
                    "montant_ht": 0.0,
                    "tva": 0.0,
                    "montant_ttc": 0.0}
               """
            },
            {
                "role": "user",
                "content": texte
            }
        ]
    )

    texte_json = reponse.choices[0].message.content.strip()

    try:
        data = json.loads(texte_json)
        facture = Facture(**data) # deballage de dictionnaire
        return facture
    except Exception as e:
        print(f"Erreur parsing : {e}")
        print(f"Reponse brute : {texte_json}")
        return None
    
factures = [
    "Facture N°2024-089 du 15 mars 2024, client Martin Dupont, montant HT 1250€, TVA 20%, total TTC 1500€.",
    "Facture F-2024-442, émise le 3 juin 2024 pour la société TechCorp, HT : 890€, TVA : 178€, TTC : 1068€.",
    "Reçu N°REC-007 du 22 janvier 2024, acheteur Sophie Lambert, sous-total 340€ hors taxes, TVA 68€, total 408€."
]

print("--- Extracteur de factures ---\n")
for texte in factures:
    print(f"Texte : {texte}")
    facture = extraire_facture(texte)
    if facture:
        print(f"Numéro  : {facture.numero}")
        print(f"Date    : {facture.date}")
        print(f"Client  : {facture.client}")
        print(f"HT      : {facture.montant_ht}€")
        print(f"TVA     : {facture.tva}€")
        print(f"TTC     : {facture.montant_ttc}€")
    print("-" * 50)