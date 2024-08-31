from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# Configuration de l'API OpenAI avec votre clé API
llm = OpenAI(api_key="sk-XXXXXXXX")

# Création d'un modèle de prompt avec LangChain
template = """
Vous êtes un expert en histoire.
Expliquez pourquoi {event} a eu lieu.
"""

# Remplissage du template avec une variable
prompt = PromptTemplate(
    input_variables=["event"],
    template=template,
)


# Génération d'une réponse en fournissant l'événement spécifique

llm_chain = LLMChain(llm=llm, prompt=prompt)
response = llm_chain.run(event="la Révolution française")
print(response)
