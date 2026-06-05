import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title = "Assistant Pdf",
    page_icon="📄",
    layout="centered"
)

@st.cache_resource
def charger_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def charger_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

model = charger_model()
groq_client = charger_groq()

def lire_pdf(fichier) -> str:
    reader = PdfReader(fichier)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text()
    return texte

def decouper_en_chunks(texte: str, taille: int = 300, overlap: int = 50) -> list:
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        chunk = texte[debut:fin]
        if chunk.strip():
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

def indexer_pdf(chunks: list):
    embeddings = model.encode(chunks)
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, chunks

def rag(question: str, index, chunks: list) -> str:
    question_embedding = model.encode([question])
    question_np = np.array(question_embedding).astype('float32')
    
    distances, indices = index.search(question_np, k=3)
    chunks_pertinents = [chunks[i] for i in indices[0]]
    contexte = "\n".join([f"- {chunk}" for chunk in chunks_pertinents])

    reponse = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """Tu es un assistant qui répond aux questions sur un document.
Réponds UNIQUEMENT en te basant sur le contenu fourni.
Si la réponse n'est pas dans le document, dis-le clairement.
Réponds en français, de manière concise et professionnelle."""
            },
            {
                "role": "user",
                "content": f"Contenu du document :\n{contexte}\n\nQuestion : {question}"
            }
        ]
    )
    return reponse.choices[0].message.content

st.title("📄 Assistant PDF")
st.markdown("Uploade un PDF et pose des questions dessus.")

pdf_uploade = st.file_uploader("Choisis un fichier PDF", type="pdf")

if pdf_uploade is not None:
    if "collection" not in st.session_state or st.session_state.get("pdf_nom") != pdf_uploade.name:
        with st.spinner("Lecture et indexation du PDF..."):
            texte = lire_pdf(pdf_uploade)
            chunks = decouper_en_chunks(texte)
            index, chunks_stockes = indexer_pdf(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks_stockes
            st.session_state.pdf_nom = pdf_uploade.name
            st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Pose une question sur le document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                reponse = rag(question, st.session_state.index, st.session_state.chunks)
            st.write(reponse)

        st.session_state.messages.append({"role": "assistant", "content": reponse})

else:
    st.info("👆 Commence par uploader un PDF")
