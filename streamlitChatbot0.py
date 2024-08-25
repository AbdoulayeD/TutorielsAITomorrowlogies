import streamlit as st
from openai import OpenAI

openai_api_key = "XXXXXXXX"  # Remplacez par votre clé API OpenAI
st.title("💬 Chatbot IA GEN ")

# Initialize session state if not already done
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider?"}]

# Display the conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Capture user input
if prompt := st.chat_input():
    client = OpenAI(api_key=openai_api_key)
    
    # Append user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Generate the response from the model
    response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    
    # Append the assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
