from fastapi import FastAPI,HTTPException,UploadFile,File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
import io
import os

load_dotenv()

app = FastAPI(title = "Assistant RAG API",version = "2.0")

model = SentenceTransformer("all-MiniLM-L6-V2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()

collections = {}

def lire_pdf(contenu : bytes) -> str:
    reader = PdfReader(io.BytesIO(contenu))
    texte = ""
    for page in reader.pages : 
        texte += page.extract_text()
    return texte

def decouper_en_chunks(texte: str,taille:int=300,overlap:int=50):
    chunks=[]
    debut = 0
    while debut < len(texte):
        chunk = texte[debut:debut + taille]
        if chunk.strip():
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

class Question(BaseModel):
    texte : str
    collection_id : str

class Reponse(BaseModel):
    question : str
    reponse : str
    chunks_utilises : list[str]

class InfoPDF(BaseModel):
    collection_id : str
    nb_chunks : int
    message : str

@app.get("/")
def accueil():
    return {"message": "Assistant RAG API v2.0", "collections_actives": len(collections)}

@app.post("/upload",response_model=InfoPDF)
async def upload_pdf(fichier : UploadFile = File(...)):
    if not fichier.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="le fichier doit etre en pdf")
    
    contenu = await fichier.read()
    texte = lire_pdf(contenu)

    if not texte.strip():
        raise HTTPException(statut_code=400, detail="le PDF doit etre vide ou illisible")
    
    chunks = decouper_en_chunks(texte)
    collection_id = fichier.filename.replace(".pdf", "").replace(" ", "_")

    collection = chroma_client.get_or_create_collection(collection_id)
    embeddings = model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    collections[collection_id] = collection

    return InfoPDF(
        collection_id=collection_id,
        nb_chunks=len(chunks),
        message=f"PDF indexé avec succès — {len(chunks)} chunks créés"
    )

@app.post("/question", response_model=Reponse)
def poser_question(question: Question):
    if question.collection_id not in collections:
        raise HTTPException(status_code=404, detail="Collection introuvable — uploadez d'abord un PDF")

    if not question.texte.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    collection = collections[question.collection_id]
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
                "content": """Tu es un assistant qui répond sur un document.
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

@app.get("/collections")
def lister_collections():
    return {"collections": list(collections.keys())}
