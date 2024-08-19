# Installer les libray openai et gradio avant d'executer 
# pip install openai gradio
from openai import OpenAI
import gradio as gr

# Configurez l'API OpenAI avec votre clé API
client = OpenAI(
    api_key="XXXXXXXX"  # Remplacez par votre clé API OpenAI
)

def generate_chatbot_response(content):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Utilisation du modèle GT4-O-Mini
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    return chat_completion.choices[0].message.content
    
    

def chat_interface(user_input):
    response = generate_chatbot_response(user_input)
    return response

iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text", title="Chatbot IA avec GT4-O-Mini")
iface.launch()


