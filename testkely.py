import httpx

def get_bitcoin_price():
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-EUR/spot"
        reponse = httpx.get(url, follow_redirects=True, timeout=5)
        reponse.raise_for_status() # Vérifie que la requête a réussi, sinon lève une exception
        data = reponse.json()
        return data["data"]["amount"]
    except httpx.TimeoutException :
        print("Erreur Bitcoin: l'api ne repond pas(timeout)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Erreur Bitcoin: statut {e.response.status_code}")
        return None
    
bitcoin_price = get_bitcoin_price()
print("--- Prix du Bitcoin ---")
if bitcoin_price is not None:
    print(f"Prix actuel du Bitcoin : {bitcoin_price} EUR")
else:
    print("Prix du Bitcoin : données indisponibles")