from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
import requests

# Configuration de l'API OpenAI avec votre clé API
llm = OpenAI(api_key="sk-XXXXXXXX")

# Fonction pour effectuer un calcul mathématique
def calculate_expression(expression):
    try:
        result = eval(expression)
        return f"Le résultat de l'expression '{expression}' est {result}."
    except Exception as e:
        return f"Erreur dans le calcul : {str(e)}"

# Fonction pour obtenir des données météorologiques d'une API externe
def get_weather(city):
    try:
        # Remplacez par votre propre clé API et l'URL de votre fournisseur de services météo
        api_key = "votre_cle_api_meteo"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if data["cod"] == 200:
            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"Le temps à {city} est actuellement '{weather_description}' avec une température de {temperature}°C."
        else:
            return f"Impossible d'obtenir les données météorologiques pour {city}."
    except Exception as e:
        return f"Erreur lors de la récupération des données météo : {str(e)}"

# Définition des outils disponibles pour l'agent
tools = [
    Tool(
        name="Calculator",
        func=calculate_expression,
        description="Effectue des calculs mathématiques simples. Exemple : 'calcul 3 + 5 * 2'"
    ),
    Tool(
        name="WeatherAPI",
        func=get_weather,
        description="Fournit des informations météorologiques pour une ville donnée. Exemple : 'météo Paris'"
    )
]

# Initialisation de l'agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description"
)

# Exemples d'utilisation de l'agent
response1 = agent.run("calcul 8 * 9")
print(response1)  # Devrait retourner le résultat du calcul

response2 = agent.run("météo Londres")
print(response2)  # Devrait retourner les informations météorologiques pour Paris

