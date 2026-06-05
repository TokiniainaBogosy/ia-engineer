from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def calculatrice(expression : str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Erreur: {str(e)}"
    
def date_du_jour() -> str:
    return datetime.now().strftime("%d/%m/%Y à %H:%M")

def chercher_info(sujet : str) -> str:
    infos = {
        "python" : "Python est créé en 1991 par Guido van Rossum.",
        "langchain": "LangChain est fondé en 2022 par Harrison Chase.",
        "groq": "Groq est une entreprise américaine fondée en 2016."
    }

    sujet_lower = sujet.lower()
    for cle,valeurs in infos.items():
        if cle in sujet_lower:
            return valeurs
    return f"Pas d'info sur : {sujet}"

outils = {
    "calculatrice": calculatrice,
    "date_du_jour": date_du_jour,
    "chercher_info": chercher_info
}

tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "calculatrice",
            "description" : "Calcule une expression mathématique",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "expression" : {"type": "string", "description": "L'expression à calculer"}
                },
                "required" : ["expression"]
            }
        }
        
    },
    {
        "type": "function",
        "function": {
            "name": "date_du_jour",
            "description": "Retourne la date et l'heure actuelles",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chercher_info",
            "description": "Cherche une info sur python, langchain ou groq",
            "parameters": {
                "type": "object",
                "properties": {
                    "sujet": {"type": "string", "description": "Le sujet à rechercher"}
                },
                "required": ["sujet"]
            }
        }
    }
]

def agent(question : str) -> str:
    print(f"\nQuestion : {question}")
    messages = [{"role": "user", "content": question}]

    for etape in range(5):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                nom = tool_call.function.name
                params = json.loads(tool_call.function.arguments)
                print(f"  → Outil : {nom}({params})")

                resultat = outils[nom](**(params or {}))
                print(f"  → Résultat : {resultat}")

                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": resultat
                    })
        else:
            print(f"Réponse : {message.content}")
            return message.content
    return "Limite atteinte"

agent("Quel heure est-il ?")
agent("Combien font 1250 * 1.20 ?")
agent("Qui a créé Python ?")
agent("Quel est le prix TTC de 890€ HT avec 20% de TVA, et quelle est la date d'aujourd'hui ?")

