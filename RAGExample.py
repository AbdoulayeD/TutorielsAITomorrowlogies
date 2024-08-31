from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# Configuration de l'API OpenAI avec votre clé API
llm = OpenAI(api_key="sk-XXXXXXXX")

# Charger des documents à partir d'un texte ou d'un corpus
loader = TextLoader("document.txt")
documents = loader.load()

# Créer un vecteur d'embeddings pour les documents
embeddings = OpenAIEmbeddings(api_key="sk-XXXXXXXX")
vector_store = FAISS.from_documents(documents, embeddings)

# Correct Usage for RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# Question utilisateur
question = "Quels sont les principaux types de macronutriments et leurs sources alimentaires ?"

# Génération de réponse augmentée par la récupération d'information
response = rag_chain.run(question)
print(response)
