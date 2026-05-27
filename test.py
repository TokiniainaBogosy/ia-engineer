from dotenv import load_dotenv #lit le fichier .env et charge les variables d'environnement
import httpx #permet de faire des requêtes HTTP, comme une alternative à requests, mais avec des fonctionnalités asynchrones et une meilleure performance
import os #permet d'interagir avec le système d'exploitation, notamment pour accéder aux variables d'environnement

load_dotenv() # lit le fichier .env et charge les variables d'environnement en memoire
print("✓ dotenv OK")

r = httpx.get("https://api.github.com")
print(f"✓ httpx OK — status {r.status_code}")

print("✓ Environnement prêt !")