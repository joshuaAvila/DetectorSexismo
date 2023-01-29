import tweepy as tw
import streamlit as st
import pandas as pd
import torch
import numpy as np
import regex as re
import pysentimiento
import geopy

from pysentimiento.preprocessing import preprocess_tweet
from geopy.geocoders import Nominatim

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AdamW
tokenizer = AutoTokenizer.from_pretrained('hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021')
model = AutoModelForSequenceClassification.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021")

import torch
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
    #text=text.lower()
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





def analizar_tweets(search_words, number_of_tweets ):
  tweets = api.user_timeline(screen_name = search_words, count= number_of_tweets)
  tweet_list = [i.text for i in tweets]
  text= pd.DataFrame(tweet_list)
  text[0] = text[0].apply(preprocess_tweet)
  text1=text[0].values
  indices1=tokenizer.batch_encode_plus(text1.tolist(), max_length=128,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)
  input_ids1=indices1["input_ids"]
  attention_masks1=indices1["attention_mask"]
  prediction_inputs1= torch.tensor(input_ids1)
  prediction_masks1 = torch.tensor(attention_masks1)
  batch_size = 25
  # Create the DataLoader.
  prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
  prediction_sampler1 = SequentialSampler(prediction_data1)
  prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)
  #print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))
  # Put model in evaluation mode
  model.eval()
  # Tracking variables 
  predictions = []
  for batch in prediction_dataloader1:
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids1, b_input_mask1 = batch

    #Telling the model not to compute or store gradients, saving memory and   # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs1 = model(b_input_ids1, token_type_ids=None,attention_mask=b_input_mask1)
    logits1 = outputs1[0]
    # Move logits and labels to CPU
    logits1 = logits1.detach().cpu().numpy()
    # Store predictions and true labels
    predictions.append(logits1)
      
  #flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = [item for sublist in predictions for item in sublist]
  
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  
  probability = np.amax(logits1,axis=1).flatten()
  Tweets =['Últimos '+ str(number_of_tweets)+' Tweets'+' de '+search_words]
  df = pd.DataFrame(list(zip(text1, flat_predictions,probability)), columns = ['Tweets' , 'Prediccion','Probabilidad'])
  
  df['Prediccion']= np.where(df['Prediccion']== 0, 'No Sexista', 'Sexista')
  df['Tweets'] = df['Tweets'].str.replace('RT|@', '')   
  #df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'[:;][-o^]?[)\]DpP3]|[(/\\]|[\U0001f600-\U0001f64f]|[\U0001f300-\U0001f5ff]|[\U0001f680-\U0001f6ff]|[\U0001f1e0-\U0001f1ff]','', x))

  tabla = st.table(df.reset_index(drop=True).head(30).style.applymap(color_survived, subset=['Prediccion']))

  return tabla

def analizar_frase(frase):
  #palabra = frase.split()
  palabra = [frase]
  
  indices1=tokenizer.batch_encode_plus(palabra,max_length=128,add_special_tokens=True, 
                                         return_attention_mask=True,
                                         pad_to_max_length=True,
                                         truncation=True)
  input_ids1=indices1["input_ids"]
  attention_masks1=indices1["attention_mask"]
  prediction_inputs1= torch.tensor(input_ids1)
  prediction_masks1 = torch.tensor(attention_masks1)
  batch_size = 25
  prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
  prediction_sampler1 = SequentialSampler(prediction_data1)
  prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)
  model.eval()
  predictions = []
  # Predict 
  for batch in prediction_dataloader1:
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids1, b_input_mask1 = batch
    # Telling the model not to compute or store gradients, saving memory and   # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs1 = model(b_input_ids1, token_type_ids=None,attention_mask=b_input_mask1)
    logits1 = outputs1[0]
    # Move logits and labels to CPU
    logits1 = logits1.detach().cpu().numpy()
    # Store predictions and true labels
    predictions.append(logits1)
  flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  tokens = tokenizer.tokenize(frase)
  # Convertir los tokens a un formato compatible con el modelo
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  attention_masks = [1] * len(input_ids)
 
  # Pasar los tokens al modelo
  outputs = model(torch.tensor([input_ids]), token_type_ids=None, attention_mask=torch.tensor([attention_masks]))
  scores = outputs[0]
  #prediccion = scores.argmax(dim=1).item()
  # Obtener la probabilidad de que la frase sea "sexista"
  probabilidad_sexista = scores.amax(dim=1).item()
  #print(probabilidad_sexista)
  
  # Crear un Dataframe
  text= pd.DataFrame({'Frase': [frase], 'Prediccion':[flat_predictions], 'Probabilidad':[probabilidad_sexista]})
  text['Prediccion'] = np.where(text['Prediccion'] == 0 , 'No Sexista', 'Sexista')


  tabla = st.table(text.reset_index(drop=True).head(30).style.applymap(color_survived, subset=['Prediccion']))
    
  return tabla

def tweets_localidad(buscar_localidad):
  geolocator = Nominatim(user_agent="nombre_del_usuario")
  location = geolocator.geocode(buscar_localidad)
  radius = "200km"
  tweets = api.search(lang="es",geocode=f"{location.latitude},{location.longitude},{radius}", count = 50)
  #for tweet in tweets:
  #  print(tweet.text)
  tweet_list = [i.text for i in tweets]
  text= pd.DataFrame(tweet_list)
  text[0] = text[0].apply(preprocess_tweet)
  text1=text[0].values
  print(text1)
  indices1=tokenizer.batch_encode_plus(text1.tolist(), max_length=128,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)
  input_ids1=indices1["input_ids"]
  attention_masks1=indices1["attention_mask"]
  prediction_inputs1= torch.tensor(input_ids1)
  prediction_masks1 = torch.tensor(attention_masks1)
  batch_size = 25
  # Create the DataLoader.
  prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
  prediction_sampler1 = SequentialSampler(prediction_data1)
  prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)
  #print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))
  # Put model in evaluation mode
  model.eval()
  # Tracking variables 
  predictions = []
  for batch in prediction_dataloader1:
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids1, b_input_mask1 = batch

    #Telling the model not to compute or store gradients, saving memory and   # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs1 = model(b_input_ids1, token_type_ids=None,attention_mask=b_input_mask1)
    logits1 = outputs1[0]
    # Move logits and labels to CPU
    logits1 = logits1.detach().cpu().numpy()
    # Store predictions and true labels
    predictions.append(logits1)
      
  #flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = [item for sublist in predictions for item in sublist]
  
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  
  probability = np.amax(logits1,axis=1).flatten()
  Tweets =['Últimos 50 Tweets'+' de '+ buscar_localidad]
  df = pd.DataFrame(list(zip(text1, flat_predictions,probability)), columns = ['Tweets' , 'Prediccion','Probabilidad'])
  
  df['Prediccion']= np.where(df['Prediccion']== 0, 'No Sexista', 'Sexista')
  #df['Tweets'] = df['Tweets'].str.replace('RT|@', '')
  #df_filtrado = df[df["Sexista"] == 'Sexista']
  #df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'[:;][-o^]?[)\]DpP3]|[(/\\]|[\U0001f600-\U0001f64f]|[\U0001f300-\U0001f5ff]|[\U0001f680-\U0001f6ff]|[\U0001f1e0-\U0001f1ff]','', x))
  
  tabla = st.table(df.reset_index(drop=True).head(50).style.applymap(color_survived, subset=['Prediccion']))
    
  df_sexista = df[df['Sexista']=="Sexista"]
  df_no_sexista = df[df['Probabilidad'] > 0]
  sexista = len(df_sexista)
  no_sexista = len(df_no_sexista)

  # Crear un gráfico de barras
  labels = ['Sexista  ', ' No sexista']
  counts = [sexista, no_sexista]
  plt.bar(labels, counts)
  plt.xlabel('Categoría')
  plt.ylabel('Cantidad de tweets')
  plt.title('Cantidad de tweets sexistas y no sexistas')
  plt.show()

  return df



    
def run():   
 with st.form("my_form"):
   col,buff1, buff2 = st.columns([2,2,1])
   st.write("Escoja una Opción")
   search_words = col.text_input("Introduzca el termino, usuario o localidad para analizar y pulse el check correspondiente")
   number_of_tweets = col.number_input('Introduzca número de tweets a analizar. Máximo 50', 0,50,10)
   termino=st.checkbox('Término')
   usuario=st.checkbox('Usuario')
   localidad=st.checkbox('Localidad')
   submit_button = col.form_submit_button(label='Analizar')
   error =False
     
   if submit_button:
            # Condición para el caso de que esten dos check seleccionados
            if ( termino == False and usuario == False and localidad == False):
                st.text('Error no se ha seleccionado ningun check')
                error=True
            elif ( termino == True and usuario == True and localidad == True):
                st.text('Error se han seleccionado varios check')
                error=True
                
            if (error == False):
                if (termino):
                  analizar_frase(search_words)
                    
                elif (usuario):
                    analizar_tweets(search_words,number_of_tweets)
                elif (localidad):
                    tweets_localidad(search_words)
                
run()
