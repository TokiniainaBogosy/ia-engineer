from sentence_transformers import SentenceTransformer # la librairie qui transforme les textes en vecteurs
from sklearn.metrics.pairwise import cosine_similarity # la fonction qui calcule la distance entre les vecteurs
import numpy as np # librairie de calcul mathematique, utilisé pour manipuler les tableaux de vecteurs

model = SentenceTransformer("all-MiniLM-L6-v2") # un modèle pré-entraîné qui génère des vecteurs de 384 dimensions à partir de phrases en anglais (fonctionne aussi pour le français)

phrases = [
    "J'aime programmer en Python",
    "Le développement en Python est passionnant",
    "La cuisine française est délicieuse",
    "J'adore coder des applications",
    "Le camembert est un fromage français"
]

print("Calcul des embeddings...")
embeddings = model.encode(phrases) # transforme chaque phrase en un vecteur de 384 dimensions, résultat : un tableau de 5 lignes (une par phrase) et 384 colonnes (une par dimension du vecteur)
print(f"Dimension d'un vecteur : {embeddings.shape[1]}\n")

query = "Python et la programmation"
query_embedding = model.encode([query]) # transforme la requête en un vecteur de 384 dimensions, résultat : un tableau de 1 ligne et 384 colonnes

similarites = cosine_similarity(query_embedding, embeddings)[0] # On mesure la distance entre le vecteur de la question et chaque vecteur de phrase. Le résultat est un score entre 0 et 1 — plus c'est proche de 1, plus les textes sont similaires.  

resultats = sorted(zip(similarites, phrases), reverse=True) # On associe chaque score de similarité à sa phrase d'origine, puis on trie les résultats du plus similaire au moins similaire.

print(f"Query : '{query}'\n")
print("Phrases les plus similaires :")
for score, phrase in resultats:
    barre = "█" * int(score * 20)
    print(f"{score:.3f} {barre} {phrase}")