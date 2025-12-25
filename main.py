import numpy as np
import pandas as pd
import streamlit as st
import pickle

#load the model
model = pickle.load(open('rf_model.pkl', 'rb'))
st.title('Bank Note Authentication Prediction')

#input features from user
variance = st.text_input('Variance')
skewness = st.text_input('Skewness')
curtosis = st.text_input('Curtosis')
entropy = st.text_input('Entropy')
output = ''
#prediction function
def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return prediction   
#button for prediction
if st.button('Predict'):
    output = predict_note_authentication(variance, skewness, curtosis, entropy)
    if output[0]==0:
        st.success('The bank note is Authentic')
    else:
        st.success('The bank note is Not Authentic')
        