import time
import streamlit as st
import requests

# Set Backend LLM API Server IP
api_url = 'http://127.0.0.1:8000'

st.title('''Welcome to Chat CPT's World!''')

# Use Instead of DB.
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streaming Effects
def stream_data(tokens):
    for word in tokens.split(" "):
        yield word + " "
        time.sleep(0.02)

# Write The Chat Logs In Container
def view_chat_logs():
    for i in range(0, len(st.session_state.chat_history), 2):
        messages.chat_message("user").write(f'{st.session_state.chat_history[i]}')
        messages.chat_message("assistant").write_stream(stream_data(f"Echo: {st.session_state.chat_history[i+1][1:-1]}")) # write_stream 글자 입력하는 이펙트

# Side Bar
with st.sidebar:
    st.title("Prompt Setting")
    '''### Top K'''
    top_k = st.slider("top_k", 0, 100, 50, label_visibility="collapsed")
    '''### Top P'''
    top_p = st.slider('top_p', 0, 100, 70, label_visibility="collapsed")
    '''### Max Token'''
    max_token = st.slider('max_token', 0, 1000, 100, label_visibility="collapsed")
    
    # Positioning Side By Side
    col1, col2 = st.columns(2)
    with col1:
        on = st.toggle('Promping')
    with col2:
        do_sample = st.checkbox('Do Sample')

    if on:
        pre_prompt = st.text_input('pre_prompt', 'Enter the Contents..', label_visibility="collapsed", max_chars=int(max_token/2))
        st.session_state['pre_prompt'] = pre_prompt  # Store the current input to session state
    else:
        st.session_state['pre_prompt'] = ""

    # Setting Path API Trigger
    if st.button('Save'):
        data = {
            'pre_prompt' : st.session_state.get('pre_prompt'),
            'do_sample' : bool(do_sample),
            'top_k': top_k,
            'top_p': float(top_p/100),
            'max_token': max_token
        }
        with st.spinner('Wait for it...'):
            response = requests.post(api_url+'/setting/', json=data) # API
        if response.status_code == 200:
            st.success('Done!')
            


# Chat Room
prompt = st.chat_input("Say something")
messages = st.container(height=800) # 각 화면에 맞게 조정

if prompt:
    st.session_state.chat_history.append(prompt)
    with st.spinner('Wait for it...'):
        response = requests.post(api_url+'/question/', params={'prompt' : prompt}) # API
    st.success('Done!')
    st.session_state.chat_history.append(response.text)
    view_chat_logs()