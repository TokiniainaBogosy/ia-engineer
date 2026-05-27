from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

class Produit(BaseModel):
    nom: str = Field(description="Nom du produit")
    prix: float = Field(description="Prix en euros")
    categorie: str = Field(description="Catégorie du produit")
    disponible: bool = Field(description="True en stock, False sinon")

parser = JsonOutputParser(pydantic_object=Produit)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un extracteur de données produit.
Extrais les informations du texte et réponds UNIQUEMENT en JSON valide sans texte avant ou après.
Format attendu avec ces clés EXACTES en français
{format_instructions}"""),
    ("human", "{texte}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

produits = [
    "Le iPhone 15 Pro est disponible en stock au prix de 1199 euros, c'est un smartphone.",
    "Rupture de stock sur le vélo électrique Trek à 2499€, catégorie sport et loisirs.",
    "Casque Sony WH-1000XM5 en stock, 279,99€, catégorie audio."
]

print("--- Extracteur de produits ---\n")
for texte in produits:
    print(f"Texte : {texte}")
    produit = chain.invoke({"texte": texte})
    print(f"Nom        : {produit['nom']}")
    print(f"Prix       : {produit['prix']}€")
    print(f"Catégorie  : {produit['categorie']}")
    print(f"Disponible : {'✅' if produit['disponible'] else '❌'}")
    print("-" * 50)