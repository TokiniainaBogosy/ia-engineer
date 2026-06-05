from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def calculatrice(expression: str) -> str:
    try:
        resultat = eval(expression)
        return str(resultat)
    except Exception as e:
        return f"Erreur : {e}"
    
def date_du_jour() -> str:
    return datetime.now().strftime("%d/%m/%Y à %H:%M")

def chercher_info(sujet: str) -> str:
    infos = {
        "python": "Python est créé en 1991 par Guido van Rossum.",
        "langchain": "LangChain est fondé en 2022 par Harrison Chase.",
        "groq": "Groq est une entreprise américaine fondée en 2016."
    }
    sujet_lower = sujet.lower()
    for cle, valeur in infos.items():
        if cle in sujet_lower:
            return valeur
    return f"Pas d'info sur : {sujet}"

outils = {
    "calculatrice": calculatrice,
    "date_du_jour": date_du_jour,
    "chercher_info": chercher_info
}

system_prompt = """Tu es un agent IA avec 3 outils disponibles.

OUTILS DISPONIBLES :
- calculatrice : calcule une expression mathématique
- date_du_jour : retourne la date et l'heure actuelles  
- chercher_info : cherche une info sur python, langchain ou groq

RÈGLES STRICTES :
1. Réponds TOUJOURS avec UN SEUL objet JSON valide
2. Le champ "outil" contient UNIQUEMENT le nom exact : calculatrice, date_du_jour, ou chercher_info
3. Jamais de texte avant ou après le JSON
4. Quand tu as la réponse, utilise reponse_finale

FORMAT pour utiliser un outil :
{"outil": "calculatrice", "params": {"expression": "2+2"}}
{"outil": "date_du_jour", "params": {}}
{"outil": "chercher_info", "params": {"sujet": "python"}}

FORMAT pour répondre :
{"reponse_finale": "ta réponse ici"}

Si la question demande plusieurs choses, traite-les une par une dans l'ordre."""

def agent(question : str) -> str:
    print(f"\nQuestion : {question}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for etape in range(5): # Limite à 5 étapes pour éviter les boucles infinies
        reponse = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )

        contenu = reponse.choices[0].message.content.strip()

        try:
            data = json.loads(contenu)

            if "reponse_finale" in data:
                print(f"Réponse finale : {data['reponse_finale']}")
                return data["reponse_finale"]

            if "outil" in data:
                nom_outil = data["outil"]
                params = data.get("params", {})
                print(f"  → Outil : {nom_outil}({params})")

                fonction = outils[nom_outil]
                resultat = fonction(**params)
                print(f"  → Résultat : {resultat}")

                messages.append({"role": "assistant", "content": contenu})
                messages.append({"role": "user", "content": f"Résultat de {nom_outil} : {resultat}"})

        except json.JSONDecodeError:
            print(f"  → Réponse non JSON : {contenu}")
            return contenu

    return "Limite d'étapes atteinte"

agent("Quelle heure est-il ?")
agent("Combien font 1250 * 1.20 ?")
agent("Qui a créé Python et en quelle année ?")
agent("Quel est le prix TTC de 890€ HT avec 20% de TVA, et quelle est la date d'aujourd'hui ?")

        