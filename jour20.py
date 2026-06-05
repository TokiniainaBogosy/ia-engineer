from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Assistant RAG API", version="1.0")

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_fastapi")

documents = [
    "Pour résilier un contrat, envoyez une lettre recommandée avec 30 jours de préavis.",
    "Le remboursement est possible dans les 14 jours suivant l'achat.",
    "La garantie produit est valable 2 ans à compter de la date d'achat.",
    "Le support client est disponible du lundi au vendredi de 9h à 18h.",
    "La livraison standard est gratuite pour toute commande supérieure à 50€.",
    "Le paiement en 3 fois est disponible pour tout achat supérieur à 300€."
]

collection = chroma_client.get_or_create_collection("faq")
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

class Question(BaseModel):
    texte: str

class Reponse(BaseModel):
    question: str
    reponse: str
    chunks_utilises: list[str]

@app.get("/")
def accueil():
    return {"message":"Assistant RAG API — en ligne",
            "version": "1.0"
    }

@app.get("/health")
def health():
    return  {"status" : "ok"}

@app.post("/question", response_model=Reponse)
def poser_question(question: Question):
    if not question.texte.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    question_embedding = model.encode([question.texte]).tolist()  
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
Réponds en français, de manière concise."""
            },
            {
                "role": "user",
                "content": f"Extraits :\n{contexte}\n\nQuestion : {question.texte}"
            }
        ]
    )

    return Reponse(
        question=question.texte,
        reponse=reponse.choices[0].message.content,
        chunks_utilises=chunks
    ) 
