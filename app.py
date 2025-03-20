import pickle
import streamlit as st
import requests
import pandas as pd

st.header('Spam Email Detection')
input_sms=st.text_input('Enter the Message')
if st.button('Predict'):
    vectorizer=pickle.load(open('vectorizer.pkl','rb'))
    model=pickle.load(open('model.pkl','rb'))
    vectorized_input=vectorizer.transform([input_sms])
    result=model.predict(vectorized_input)[0]
    if result==1:
       st.header("Spam")
    else:
       st.header("Not Spam")