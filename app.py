import streamlit as st
import pandas as pd
import torch
import numpy as np
import re
from pysentimiento.preprocessing import preprocess_tweet

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AdamW
tokenizer = AutoTokenizer.from_pretrained('hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021')
model = AutoModelForSequenceClassification.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021")


if torch.cuda.is_available():  
    device = torch.device(	"cuda")
    print('I will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

    
consumer_key = "BjipwQslVG4vBdy4qK318KnoA"
consumer_secret = "3fzL70v9faklrPgvTi3zbofw9rwk92fgGdtAslFkFYt8kGmqBJ"
access_token = "1217853705086799872-Y5zEChpTeKccuLY3XJRXDPPZhNrlba"
access_token_secret = "pqQ5aFSJxzJ2xnI6yhVtNjQO36FOu8DBOH6DtUrPAU54J"
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def preprocess(text):
    text=text.lower()
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
    #Replace &amp, &lt, &gt with &,<,> respectively
    text=text.replace(r'&amp;?',r'and')
    text=text.replace(r'&lt;',r'<')
    text=text.replace(r'&gt;',r'>')
    #remove hashtag sign
    #text=re.sub(r"#","",text)   
    #remove mentions
    text = re.sub(r"(?:\@)\w+", '', text)
    #text=re.sub(r"@","",text)
    #remove non ascii chars
    text=text.encode("ascii",errors="ignore").decode()
    #remove some puncts (except . ! ?)
    text=re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+','',text)
    text=re.sub(r'[!]+','!',text)
    text=re.sub(r'[?]+','?',text)
    text=re.sub(r'[.]+','.',text)
    text=re.sub(r"'","",text)
    text=re.sub(r"\(","",text)
    text=re.sub(r"\)","",text)
    text=" ".join(text.split())
    return text

    
def highlight_survived(s):
    return ['background-color: red']*len(s) if (s.Sexista == 1) else ['background-color: green']*len(s)

def color_survived(val):
    color = 'red' if val=='Sexista' else 'white'
    return f'background-color: {color}'                

st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

#background-color: Blue;

colT1,colT2 = st.columns([2,8])
with colT2:
    #st.title('Analisis de comentarios sexistas en Twitter') 
    st.markdown(""" <style> .font {
    font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Análisis de comentarios sexistas en Twitter</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:28px ; font-family: 'Times New Roman'; color: #8d33ff;} 
    </style> """, unsafe_allow_html=True)

    st.markdown(""" <style> .font2 {
    font-size:16px ; font-family: 'Times New Roman'; color: #3358ff;} 
    </style> """, unsafe_allow_html=True)

