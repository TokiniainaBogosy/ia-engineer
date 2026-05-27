import httpx 

def get_meteo():
    try:
        url ="https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "Europe/Paris"
        }
        reponse = httpx.get(url, params=params,follow_redirects=True,timeout=5)
        reponse.raise_for_status() # Vérifie que la requête a réussi, sinon lève une exception
        data = reponse.json()

        return data["current"]["temperature_2m"]
    except httpx.TimeoutException :
        print("Erreur meteo: l'api ne repond pas(timeout)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Erreur meteo: statut {e.response.status_code}")
        return None


def get_taux_change():
    try:
        url = "https://api.frankfurter.app/latest"
        params = {
            "from": "EUR",
            "to": "USD"
        }
        reponse = httpx.get(url, params=params,follow_redirects=True,timeout=5)
        reponse.raise_for_status() # Vérifie que la requête a réussi, sinon lève une exception
        data = reponse.json()
        return data["rates"]["USD"]
    except httpx.TimeoutException :
        print("Erreur taux de change: l'api ne repond pas(timeout)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Erreur taux de change: statut {e.response.status_code}")
        return None

temperature = get_meteo()
taux_change = get_taux_change()

print("--- Résumé du jour ---")

if temperature is not None:
    print(f"Température à Paris : {temperature}°C")
else:
    print("Température : données indisponibles")

if taux_change is not None:
    print(f"Taux EUR/USD : {taux_change}")
else:
    print("Taux EUR/USD : données indisponibles")