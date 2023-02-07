import tweepy as tw
import streamlit as st
import pandas as pd
import regex as re
import numpy as np
import pysentimiento
import geopy
import matplotlib.pyplot as plt


from pysentimiento.preprocessing import preprocess_tweet
from geopy.geocoders import Nominatim
from transformers import pipeline


model_checkpoint = "hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021" 
pipeline_nlp = pipeline("text-classification", model=model_checkpoint)

    
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

def analizar_tweets(search_words, number_of_tweets):
  tabla = []
  if(number_of_tweets > 0 and search_words != "" ):
      try:
          # Buscar la información del perfil de usuario
          user = api.get_user(screen_name=search_words)
          #st.text(f"La cuenta {search_words} existe.")
          tweets = api.user_timeline(screen_name = search_words,tweet_mode="extended", count= number_of_tweets)
          result = []
          for tweet in tweets:
              if (tweet.full_text.startswith('RT')):
                  continue
              else:
                  datos = preprocess(tweet.full_text)
                  if datos == "":
                      continue 
                  else:
                      prediction = pipeline_nlp(datos)
                      for predic in prediction:
                          etiqueta = {'Tweets': datos,'Prediccion': predic['label'], 'Probabilidad': predic['score']}
                          result.append(etiqueta)         
          df = pd.DataFrame(result)
          df['Prediccion'] = np.where( df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
          df = df[df["Prediccion"] == 'Sexista']
          df = df[df["Probabilidad"] > 0.5]
          if df.empty:
              muestra= st.text("No hay tweets a analizar")
              tabla.append(muestra)
          else:
              muestra = st.table(df.reset_index(drop=True).head(30).style.applymap(color_survived, subset=['Prediccion']))
              tabla.append(muestra)
              #resultado=df.groupby('Prediccion')['Probabilidad'].sum()
              #colores=["#aae977","#EE3555"]
              #fig, ax = plt.subplots(figsize=(2, 1), subplotpars=None)
              #plt.pie(resultado,labels=resultado.index,autopct='%1.1f%%',colors=colores)
              #ax.set_title("Porcentajes por Categorias", fontsize=2, fontweight="bold")
              #plt.rcParams.update({'font.size':2, 'font.weight':'bold'})
              #ax.legend()
              # Muestra el gráfico
              #plt.show()
              #st.set_option('deprecation.showPyplotGlobalUse', False)
              #st.pyplot()
      except Exception as e:
          muestra = st.text(f"La cuenta {search_words} no existe.") 
          tabla.append(muestra) 
  else:
      muestra= st.text("Ingrese los parametros correspondientes")
      tabla.append(muestra)
        
  return tabla

def tweets_localidad(buscar_localidad):
    tabla = []
    try:
        geolocator = Nominatim(user_agent="nombre_del_usuario")
        location = geolocator.geocode(buscar_localidad)
        radius = "15km"
        tweets = api.search_tweets(q="",lang="es",geocode=f"{location.latitude},{location.longitude},{radius}", count = 1000, tweet_mode="extended")
        result = []
        for tweet in tweets:
            if (tweet.full_text.startswith('RT')):
                continue
            elif not tweet.full_text.strip():
                continue
            else:
                datos = preprocess(tweet.full_text)
                prediction = pipeline_nlp(datos)
                for predic in prediction:
                    etiqueta = {'Tweets': datos,'Prediccion': predic['label'], 'Probabilidad': predic['score']}
                    result.append(etiqueta)
        df = pd.DataFrame(result)
        
        #muestra = st.table(df.reset_index(drop=True).head(5).style.applymap(color_survived, subset=['Prediccion']))
        
        if df.empty:
            muestra=st.text("No se encontraron tweets sexistas dentro de la localidad")
            tabla.append(muestra)
        else:
            #tabla.append(muestra)
            df.sort_values(by=['Prediccion', 'Probabilidad'], ascending=[False, False], inplace=True)
            df['Prediccion'] = np.where(df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
            #df = df[df["Prediccion"] == 'Sexista']
            #df = df[df["Probabilidad"] > 0.5]
            #df = df.sort_values(by='Probabilidad', ascending=False)
            df['Probabilidad'] = df['Probabilidad'].apply(lambda x: round(x, 3))
            muestra = st.table(df.reset_index(drop=True).head(10).style.applymap(color_survived, subset=['Prediccion']))
            tabla.append(muestra)
            resultado=df.groupby('Prediccion')['Probabilidad'].mean()
            colores=["#EE3555","#aae977"]
            fig, ax = plt.subplots()
            fig.set_size_inches(2, 2)
            plt.pie(resultado,labels=resultado.index,autopct='%1.1f%%',colors=colores,  textprops={'fontsize': 4})
            ax.set_title("Porcentajes por Categorias", fontsize=5, fontweight="bold")
            plt.rcParams.update({'font.size':4, 'font.weight':'bold'})
            ax.legend()
            # Muestra el gráfico
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        
    except AttributeError as e:
        muestra=st.text("No existe ninguna localidad con ese nombre") 
        tabla.append(muestra)
                 
    return tabla 
    
def analizar_frase(frase):

    if frase == "":
        tabla = st.text("Ingrese una frase")
        #st.text("Ingrese una frase")
    else:
        predictions = pipeline_nlp(frase)
        # convierte las predicciones en una lista de diccionarios
        data = [{'Texto': frase, 'Prediccion': prediction['label'], 'Probabilidad': prediction['score']} for prediction in predictions]
        # crea un DataFrame a partir de la lista de diccionarios
        df = pd.DataFrame(data)
        df['Prediccion'] = np.where( df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
        # muestra el DataFrame
        tabla = st.table(df.reset_index(drop=True).head(5).style.applymap(color_survived, subset=['Prediccion']))
        
    return tabla
    
def run():   
 with st.form("my_form"):
   col,buff1, buff2 = st.columns([2,2,1])
   st.write("Escoja una Opción")
   search_words = col.text_input("Introduzca la frase, el usuario o localidad para analizar y pulse el check correspondiente")
   number_of_tweets = col.number_input('Introduzca número de tweets a analizar del usuario Máximo 50', 0,50,0)
   termino=st.checkbox('Frase')
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
