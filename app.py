
import streamlit as st
import pandas as pd
import torch
import numpy as np
import regex as re

def highlight_survived(s):
    return ['background-color: red']*len(s) if (s.Sexista == 1) else ['background-color: green']*len(s)

def color_survived(val):
    color = 'red' if val=='Sexista' else 'white'
    return f'background-color: {color}'                

st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

colT1,colT2 = st.columns([2,8])
with colT2:
   # st.title('Analisis de comentarios sexistas en Twitter') 
    st.markdown(""" <style> .font {
    font-size:40px ; font-family: 'Cooper Black'; color: #06bf69;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Análisis de comentarios sexistas en Twitter</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:28px ; font-family: 'Times New Roman'; color: #8d33ff;} 
    </style> """, unsafe_allow_html=True)

    st.markdown(""" <style> .font2 {
    font-size:16px ; font-family: 'Times New Roman'; color: #3358ff;} 
    </style> """, unsafe_allow_html=True)

   
def run():   
 with st.form("my_form"):
   col,buff1, buff2 = st.columns([2,2,1])
   st.write("Escoja una Opción")
   search_words = col.text_input("Introduzca el termino o usuario para analizar y pulse el check correspondiente")
   number_of_tweets = col.number_input('Introduzca número de twweets a analizar. Máximo 50', 0,50,10)
   termino=st.checkbox('Término')
   usuario=st.checkbox('Usuario')
   submit_button = col.form_submit_button(label='Analizar')
   error=False
   if submit_button:
            date_since = "2020-09-14"
            if ( termino == False and usuario == False):
                st.text('Error no se ha seleccionado ningun check')
                error=True
            elif ( termino == True and usuario == True):
                st.text('Error se han seleccionado los dos check')
                error=True 
      

run()
