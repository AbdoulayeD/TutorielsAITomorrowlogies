import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import os

# Configurez votre clé API OpenAI
os.environ["OPENAI_API_KEY"] = "sk-XXXXXX"

@st.cache_resource
def load_qa_chain():
    # Chargez les documents
    loader = TextLoader("boutique_db.txt")
    documents = loader.load()

    # Divisez le texte en chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Créez des embeddings
    embeddings = OpenAIEmbeddings()

    # Créez une base de données vectorielle
    db = Chroma.from_documents(texts, embeddings)

    # Créez un objet RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    return qa

# Titre de l'application
st.title("Assistant de Gestion de Boutique RAG")

# Chargez la chaîne QA
qa = load_qa_chain()

# Zone de saisie pour la question
question = st.text_input("Posez une question sur la gestion de la boutique:")

# Bouton pour soumettre la question
if st.button("Obtenir une réponse"):
    if question:
        # Obtenez la réponse
        response = qa.run(question)
        
        # Affichez la réponse
        st.write("Réponse:")
        st.write(response)
    else:
        st.write("Veuillez entrer une question.")

# Ajoutez quelques exemples de questions
st.sidebar.header("Exemples de questions:")
st.sidebar.write("1. Quel est le prix de vente des jeans et combien y en a-t-il en stock ?")
st.sidebar.write("2. Quelle est la marge bénéficiaire sur les T-shirts ?")
st.sidebar.write("3. Quel produit a le stock le plus élevé ?")
