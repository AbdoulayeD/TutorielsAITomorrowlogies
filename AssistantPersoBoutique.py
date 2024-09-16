
import os
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import tool
from langgraph.graph.message import AnyMessage, add_messages
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
import re
import streamlit as st
from whisper_stt import whisper_stt
from streamlit_extras.stylable_container import stylable_container
from langchain_core.messages import AIMessage
from openai import OpenAI
from pathlib import Path
import base64


import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import tool
# Assurez-vous d'avoir défini votre clé API OpenAI
import shutil
import uuid
from langchain.globals import set_debug
#set_debug(True)


# Assurez-vous d'avoir défini votre clé API OpenAI

openai_api_key="sk-XXXXXXXXX"

os.environ["OPENAI_API_KEY"] = openai_api_key


# Définir les types pour notre état

    
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Fonction pour créer le système RAG pour la boutique
def create_store_qa():
    loader = UnstructuredExcelLoader("./boutique_database.xls")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

# Créer les outils
search = DuckDuckGoSearchRun()
store_qa = create_store_qa()



@tool
def recherche_web(query: str) -> str:
    """
    Cet outil effectue une recherche web générale pour obtenir des informations sur divers sujets,
    y compris la météo actuelle et les événements récents.

    Args:
    - query (str): La requête de recherche.

    Returns:
    - str: Les résultats de la recherche.

    Example:
    ```
    resultat = recherche_web("Quel temps fait-il aujourd'hui à Paris ?")
    ```
    """
    return search.run(query)

@tool
def info_boutique(question: str) -> str:
    """
    Cet outil fournit des informations spécifiques sur la boutique, y compris les produits,
    les stocks, les ventes et d'autres données pertinentes.

    Args:
    - question (str): La question sur la boutique.

    Returns:
    - str: La réponse à la question sur la boutique.

    Example:
    ```
    reponse = info_boutique("Quels sont nos meilleurs produits ce mois-ci ?")
    ```
    """
    return store_qa.run(question)
    
    

    
    


@tool
def calculer_indicateurs_ventes():
    """
    Cet outil calcule plusieurs indicateurs de vente utiles pour la boutique en lisant les données depuis un fichier Excel.
    Il fournit des informations sur les ventes moyennes, les meilleures ventes, la croissance, etc.

    Returns:
    - str: Un résumé des indicateurs de vente calculés.

    Example:
    ```
    indicateurs = calculer_indicateurs_ventes()
    ```
    """
    # Charger les données depuis le fichier Excel
    df = pd.read_excel('boutique_database.xlsx')
    
    # Calculer la moyenne des ventes sur 6 mois pour chaque produit
    df['Moyenne 6 mois'] = df['Ventes (6 derniers mois)'].apply(lambda x: sum(map(int, x.split(', '))) / 6)
    
    # Calculer les indicateurs
    ventes_totales_dernier_mois = df['Ventes (dernier mois)'].sum()
    produit_plus_vendu = df.loc[df['Ventes (dernier mois)'].idxmax(), 'Produit']
    ventes_max = df['Ventes (dernier mois)'].max()
    produit_moins_vendu = df.loc[df['Ventes (dernier mois)'].idxmin(), 'Produit']
    ventes_min = df['Ventes (dernier mois)'].min()
    moyenne_ventes = df['Ventes (dernier mois)'].mean()
    
    # Calculer la croissance des ventes
    df['Ventes mois précédent'] = df['Ventes (6 derniers mois)'].apply(lambda x: int(x.split(', ')[-2]))
    df['Croissance'] = (df['Ventes (dernier mois)'] - df['Ventes mois précédent']) / df['Ventes mois précédent'] * 100
    produit_plus_forte_croissance = df.loc[df['Croissance'].idxmax(), 'Produit']
    croissance_max = df['Croissance'].max()
    
    # Calculer la marge bénéficiaire moyenne
    df['Marge'] = (df['Prix de vente (€)'] - df['Prix d\'achat (€)']) / df['Prix de vente (€)'] * 100
    marge_moyenne = df['Marge'].mean()
    
    # Préparer le résumé
    resume = f"""
    Indicateurs de vente pour la boutique :
    
    1. Ventes totales du dernier mois : {ventes_totales_dernier_mois} unités
    2. Produit le plus vendu : {produit_plus_vendu} avec {ventes_max} ventes
    3. Produit le moins vendu : {produit_moins_vendu} avec {ventes_min} ventes
    4. Moyenne des ventes par produit : {moyenne_ventes:.2f} unités
    5. Produit avec la plus forte croissance : {produit_plus_forte_croissance} (+{croissance_max:.2f}%)
    6. Marge bénéficiaire moyenne : {marge_moyenne:.2f}%
    
    Conseil : Concentrez-vous sur la promotion de {produit_plus_vendu} tout en cherchant à améliorer les ventes de {produit_moins_vendu}.
    Envisagez également d'analyser le succès de {produit_plus_forte_croissance} pour répliquer cette croissance sur d'autres produits.
    """
    
    return resume


    
    

tools = [info_boutique,calculer_indicateurs_ventes, recherche_web]

# Définir le prompt principal de l'assistant
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Vous êtes un assistante IA nommé Nova, conçu pour gérer efficacement une boutique en utilisant les outils disponibles. "
            "Vous répondez toujours poliment et votre fonction principale est d'interagir intelligemment et de traiter "
            "les informations avec précision concernant les opérations de la boutique.\n\n"
            "Vous avez accès aux outils suivants :\n{tools}\n\n"
            "Pour spécifier un outil, utilisez un blob JSON avec une clé 'action' (nom de l'outil) et une clé 'action_input' (entrée de l'outil).\n\n"
            "Valeurs 'action' valides : 'Final Answer' ou les noms des outils.\n\n"
            "Lorsque vous répondez à une question sur la boutique, suivez ce processus structuré :\n\n"
            "1. Question : Question à répondre\n"
            "2. Réflexion : Déterminez si un outil est nécessaire pour répondre à la question. Préférez toujours utiliser un outil pour les questions liées à la boutique. "
            "En cas de doute, utilisez un outil. Ne spéculez pas et ne fournissez pas d'informations au-delà de la portée des outils.\n"
            "3. Action : Fournissez UNE SEULE action par blob JSON, comme indiqué :\n\n"
            "```\n"
            "{{\n  \"action\": \"$NOM_OUTIL\",\n  \"action_input\": \"$ENTRÉE\"\n}}\n"
            "```\n\n"
            "4. Observation : Enregistrez le résultat de l'action, en vous assurant qu'il respecte le format de sortie spécifié par l'outil.\n"
            "5. Vérification : Vérifiez l'observation avec les informations connues et l'historique du chat. Si le résultat n'est pas clair ou semble incorrect, indiquez que l'information n'a pas pu être vérifiée.\n"
            "6. Réponse : Sur la base des informations vérifiées, formulez une réponse. Assurez-vous que le format de la réponse suit strictement le format de sortie spécifié par l'outil.\n\n"
            "Répétez le cycle Réflexion/Action/Observation/Vérification au maximum deux fois si nécessaire.\n\n"
            "Pour les questions non liées à la boutique, répondez directement sans utiliser d'outil uniquement si vous êtes certain de la réponse. En cas d'incertitude, indiquez que l'information n'est pas disponible.\n\n"
            "Action :\n"
            "```\n"
            "{{\n  \"action\": \"Final Answer\",\n  \"action_input\": \"Votre réponse directe ici\"\n}}\n"
            "```\n\n"
            "Commencez ! N'oubliez pas de TOUJOURS répondre avec un blob JSON valide d'une seule action. Utilisez des outils si nécessaire, vérifiez toutes les réponses et tenez compte de l'historique du chat s'il peut aider à répondre à la question.",
        ),
        ("placeholder", "{messages}"),
        ]
).partial(tools=tools)

# Fonction pour gérer les erreurs d'outils
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Erreur : {repr(error)}\n veuillez corriger vos erreurs.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Créer un nœud d'outil avec gestion des erreurs
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Initialiser le modèle
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key =openai_api_key, max_tokens=1024)

# Créer l'assistant
class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:

            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    


primary_assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)

# Créer le graphe
builder = StateGraph(State)

# Définir les nœuds
builder.add_node("assistant", Assistant(primary_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Définir les arêtes
builder.set_entry_point("assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()


# Compiler le graphe
def str_to_dict(json_str):
    # Supprimer les backticks et les espaces au début et à la fin
    json_str = json_str.strip().strip('`')
    
    try:
        # Convertir la chaîne JSON en dictionnaire
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        print(f"Erreur lors de la conversion JSON : {e}")
        return None
        
        




#Fonction pour interroger l'agent
def query_agent(question: str) -> str:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    result_content = result["messages"][-1].content
    match = re.search(r'"action_input":', result_content , re.DOTALL)
    if match:
        print("-->Found action input: ", result_content.find("action_input"))
        return result_content[result_content.rfind(":")+1: result_content.find("}")-1].replace('"', "")
    return result["messages"][-1].content.strip()

    
    
#print("----->", query_agent("Quel temps fait-il aujourd'hui à Paris ?"))
#print("----->", query_agent("Quels sont nos meilleurs produits ce mois-ci ?"))
#print("----->", query_agent("Qui a gagné la Coupe du Monde de football en 2022 ?"))
#print("----->", query_agent("Donne moi le résumé des indicateurs de ventes "))





       

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


speech_file_path = Path(__file__).parent / "speech.mp3"



client = OpenAI(api_key=openai_api_key)

st.title("💬 Assistant Personnel ")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour Abdou 🙂"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
with stylable_container(
        key="bottom_content",
        css_styles="""
            {
                position: fixed;
                bottom: 105px;
            }
            """,
    ):audio = whisper_stt(openai_api_key=openai_api_key, language = 'fr')
       
       
if prompt := st.chat_input("Ask here"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)
    message = st.session_state.messages
    try:
  
        response = query_agent(str(prompt ))
        response_str = str(response)
        if response_str == "":
            response_str = "You're request didn't succeed. Can You ask the question again ?"
        st.session_state.messages.append({"role": "assistant", "content": response_str})
        
        
        
        audio_response = client.audio.speech.create(
          model="tts-1",
          voice="nova",
          input=response_str
        )
        audio_response.stream_to_file(speech_file_path)
        
        
        st.chat_message("assistant").write(response_str[:-1])
        autoplay_audio("./speech.mp3")

    #except err:
    except Exception as e:
        st.error( f"The agent_excecutor catch an error {e}, please try again!", icon="🚨")
        #return f"An error occurred: {e}"
    
        # st.error('The agent_excecutor catch an error, please try again!', icon="🚨")
        response_str = "The agent_excecutor catch an error, please try again!"
        st.session_state.messages.append({"role": "assistant", "content": response_str})

if audio:
    st.session_state.messages.append({"role": "user", "content": audio})
    st.chat_message("user").write(audio)

    message = st.session_state.messages
    try:

        response = query_agent(str(audio))


        response_str = str(response)
        if response_str == "":
            response_str = "You're request didn't succeed. Can You ask the question again ?"
        st.session_state.messages.append({"role": "assistant", "content": response_str})
        
        
        audio_response = client.audio.speech.create(
          model="tts-1",
          voice="nova",
          input=response_str
        )
        audio_response.stream_to_file(speech_file_path)

        st.chat_message("assistant").write(response_str)
        autoplay_audio("./speech.mp3")

    except:
        st.error('The agent_excecutor catch an error, please try again!', icon="🚨")
        response_str = "The agent_excecutor catch an error, please try again!"
        st.session_state.messages.append({"role": "assistant", "content": response_str})
