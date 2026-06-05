from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_pdf")

collection = chroma_client.get_or_create_collection(name="lettre_motivation")

def lire_pdf(chemin : str) -> str:
    reader = PdfReader(chemin)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text()
    return texte

def decouper_en_chunks(texte : str , taille : int = 300, overlap : int = 50) -> list:
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        chunk = texte[debut:fin]
        if chunk.strip():
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

print("Lecture du PDF...")
texte = lire_pdf("lettre_motivation_stage_dev_web .pdf")
print(f"Texte extrait : {len(texte)} caractères\n")

chunks = decouper_en_chunks(texte)
print(f"Nombre de chunks : {len(chunks)}")

print("Indexation dans ChromaDB...")
embeddings = model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("Indexation terminée\n")

def rag(question : str) -> str:
    question_embedding = model.encode([question]).tolist()
    resultats = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    chunks_pertinents = resultats["documents"][0]
    contexte = "\n".join([f"- {chunk}" for chunk in chunks_pertinents])

    reponse = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """Tu es un assistant qui répond aux questions sur un document.
                Réponds UNIQUEMENT en te basant sur le contenu fourni.
                Si la réponse n'est pas dans le document, dis-le clairement.
                Réponds en français, de manière concise."""
            },
            {
                "role": "user",
                "content": f"""Contenu du document :
                {contexte}

                Question : {question}"""
            }
        ]
    )
    return reponse.choices[0].message.content

questions = [
    "Quel est l'objet de cette lettre ?",
    "Quelles sont les compétences mentionnées ?",
    "Quel poste est visé ?",
    "Pourquoi le candidat est-il motivé ?"
]

print("--- Questions sur le PDF ---\n")
for question in questions:
    print(f"Question : {question}")
    print(f"Réponse  : {rag(question)}\n")
    print("-" * 50)
