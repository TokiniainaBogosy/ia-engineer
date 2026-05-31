from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db") # Crée un client ChromaDB qui stocke les données dans le dossier "chroma_db"

collection = client.get_or_create_collection(name="documents") # Crée une collection nommée "documents" dans la base de données ChromaDB

documents = [
    "Python est un langage de programmation interprété et orienté objet.",
    "LangChain est un framework pour construire des applications avec des LLMs.",
    "ChromaDB est une base de données vectorielle open source.",
    "Le RAG permet de connecter un LLM à des documents externes.",
    "Les embeddings transforment du texte en vecteurs numériques.",
    "FastAPI est un framework web Python rapide et moderne.",
    "Streamlit permet de créer des interfaces web en Python facilement.",
    "Les agents IA peuvent utiliser des outils pour agir de manière autonome."
]

print("Indexation des documents...")
embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"{len(documents)} documents indexés\n")

questions = [
    "Comment créer une interface web ?",
    "C'est quoi un embedding ?",
    "Comment connecter un LLM à mes données ?"
]

print("--- Recherche dans la base ---\n")
for question in questions:
    question_embedding = model.encode([question]).tolist()

    resultats = collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )

    print(f"Question : {question}")
    for doc in resultats["documents"][0]:
        print(f"  → {doc}")
    print()