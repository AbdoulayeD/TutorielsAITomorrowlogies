from streamlit_mic_recorder import mic_recorder
import streamlit as st
import io
from openai import OpenAI
import os

def whisper_stt(openai_api_key=None, start_prompt="🎙️OFF", stop_prompt="🎙️ON", just_once=False,
               use_container_width=False, language=None, callback=None, args=(), kwargs=None, key=None):
    if not 'openai_client' in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    
    audio = mic_recorder(start_prompt=start_prompt, stop_prompt=stop_prompt, just_once=just_once,
                         use_container_width=use_container_width, format="wav", key=key)
    
    if audio is None:
        return None
    
    id = audio['id']
    if id > st.session_state._last_speech_to_text_transcript_id:
        st.session_state._last_speech_to_text_transcript_id = id
        audio_bio = io.BytesIO(audio['bytes'])
        audio_bio.name = 'audio.wav'
        success = False
        err = 0
        output = None
        
        while not success and err < 3:  # Retry up to 3 times in case of OpenAI server error.
            try:
                transcript = st.session_state.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_bio,
                    language=language
                )
            except Exception as e:
                print(str(e))  # log the exception in the terminal
                err += 1
            else:
                success = True
                output = transcript.text
        
        if callback:
            callback(*args, **(kwargs or {}))
        
        return output
    return None

# import io
# import os
# from openai import OpenAI
# from streamlit_mic_recorder import mic_recorder

# def whisper_stt(openai_api_key=None, start_prompt="🎙️OFF", stop_prompt="🎙️ON", just_once=False,
#                 use_container_width=False, language=None, key=None):
#     """
#     This function performs speech-to-text conversion using OpenAI's Whisper model and captures audio using Streamlit mic_recorder.
    
#     Args:
#         openai_api_key (str, optional): The OpenAI API key. If not provided, it will be fetched from the environment variable 'OPENAI_API_KEY'.
#         start_prompt (str, optional): Prompt to display when the recorder is off.
#         stop_prompt (str, optional): Prompt to display when the recorder is on.
#         just_once (bool, optional): If True, records audio only once.
#         use_container_width (bool, optional): If True, uses the container width for the recorder.
#         language (str, optional): The language of the audio. If not specified, the language will be detected automatically.
#         key (str, optional): An optional key for Streamlit components.
    
#     Returns:
#         str: The transcribed text from the audio, or None if no audio is recorded.
#     """
#     audio = mic_recorder(start_prompt=start_prompt, stop_prompt=stop_prompt, just_once=just_once,
#                          use_container_width=use_container_width, format="webm", key=key)
    
#     if audio is None:
#         return None
    
#     openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))

#     audio_bio = io.BytesIO(audio['bytes'])
#     audio_bio.name = 'audio.webm'
#     success = False
#     err = 0
#     output = None

#     while not success and err < 3:  # Retry up to 3 times in case of OpenAI server error.
#         try:
#             transcript = openai_client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_bio,
#                 language=language
#             )
#         except Exception as e:
#             print(str(e))  # log the exception in the terminal
#             err += 1
#         else:
#             success = True
#             output = transcript.text

#     return output
