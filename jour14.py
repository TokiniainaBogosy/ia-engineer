from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv() # Charge les variables d'environnement depuis un fichier .env

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) # Initialise le client Groq avec la clé API stockée dans les variables d'environnement
chroma_client = chromadb.PersistentClient(path="./chroma_db_rag") # Initialise le client ChromaDB pour stocker les données de RAG

collection = chroma_client.get_or_create_collection(name="base_connaissance") 

documents = [
    "Pour résilier un contrat, vous devez envoyer une lettre recommandée avec un préavis de 30 jours.",
    "Les frais de résiliation anticipée s'élèvent à 150€ pour tout contrat de moins d'un an.",
    "Le remboursement est possible dans les 14 jours suivant l'achat, sans justification.",
    "Pour contacter le support client, appelez le 01 23 45 67 89 du lundi au vendredi de 9h à 18h.",
    "La garantie produit est valable 2 ans à compter de la date d'achat.",
    "En cas de produit défectueux, le remplacement est effectué sous 5 jours ouvrés.",
    "Le paiement en 3 fois sans frais est disponible pour tout achat supérieur à 300€.",
    "La livraison standard est gratuite pour toute commande supérieure à 50€."
]

print("Indexation des documents...")
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))] # ????
)

print(f"{len(documents)} documents indexés\n")

def rag(question):
    question_embedding = model.encode([question]).tolist()
    resultats = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    chunks = resultats["documents"][0]
    contexte = "\n".join([f"- {chunk}" for chunk in chunks])

    reponse = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """Tu es un assistant service client.
                Réponds UNIQUEMENT en te basant sur les extraits fournis.
                Si la réponse n'est pas dans les extraits, dis-le clairement.
                Réponds en français, de manière concise et professionnelle."""
            },
            {
                "role": "user",
                "content": f"""Extraits pertinents :
                {contexte}

                Question : {question}"""
            }
        ]
    )
    return reponse.choices[0].message.content

questions = [
    "Comment résilier mon contrat ?",
    "Quel est le délai de remboursement ?",
    "Puis-je payer en plusieurs fois ?",
    "Comment fonctionne la livraison ?"
]

print("--- Assistant RAG ---\n")
for question in questions:
    print(f"Question : {question}")
    print(f"Réponse  : {rag(question)}\n")
    print("-" * 50)
print(rag("Quel est le meilleur restaurant à Paris ?"))