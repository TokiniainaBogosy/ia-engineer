import httpx

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "current": "temperature_2m,wind_speed_10m,precipitation",
    "timezone": "Europe/Paris"
}

reponse = httpx.get(url, params=params)
data = reponse.json()

meteo = data["current"]

temperature = meteo["temperature_2m"]
vent = meteo["wind_speed_10m"]
pluie = meteo["precipitation"]
heure = meteo["time"]

print("--- Météo à Paris ---")
print(f"Heure : {heure}")
print(f"Température : {temperature}°C")
print(f"Vent : {vent} km/h")
print(f"Pluie : {pluie} mm")

if temperature > 25:
    print("Il fait chaud, n'oublie pas de boire de l'eau !")
elif temperature > 15:
    print("Temperature agréable, profite de ta journée !")
else:
    print("Il fait frais, pense à prendre une veste !")