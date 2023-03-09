import tweepy as tw
import streamlit as st
import pandas as pd
import regex as re
import numpy as np
import pysentimiento
import geopy
import matplotlib.pyplot as plt
import langdetect


from pysentimiento.preprocessing import preprocess_tweet
from geopy.geocoders import Nominatim
from transformers import pipeline
from langdetect import detect


model_checkpoint = "hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021" 
pipeline_nlp = pipeline("text-classification", model=model_checkpoint)

    
consumer_key = "BjipwQslVG4vBdy4qK318KnoA"
consumer_secret = "3fzL70v9faklrPgvTi3zbofw9rwk92fgGdtAslFkFYt8kGmqBJ"
access_token = "1217853705086799872-Y5zEChpTeKccuLY3XJRXDPPZhNrlba"
access_token_secret = "pqQ5aFSJxzJ2xnI6yhVtNjQO36FOu8DBOH6DtUrPAU54J"
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
   
def limpieza_datos(tweet):
    # Eliminar emojis
    tweet = re.sub(r'[\U0001F600-\U0001F64F]', '', tweet)
    tweet = re.sub(r'[\U0001F300-\U0001F5FF]', '', tweet)
    tweet = re.sub(r'[\U0001F680-\U0001F6FF]', '', tweet)
    tweet = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', tweet)
    # Eliminar arrobas
    tweet = re.sub(r'@\w+', '', tweet)
    # Eliminar URL
    tweet = re.sub(r'http\S+', '', tweet)
    # Eliminar hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    # Eliminar caracteres especiales
    #tweet = re.sub(r'[^a-zA-Z0-9 \n\.]', '', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9 \n\áéíóúÁÉÍÓÚñÑ.]', '', tweet)
    return tweet

def highlight_survived(s):
    return ['background-color: red']*len(s) if (s.Sexista == 1) else ['background-color: green']*len(s)

def color_survived(val):
    color = 'red' if val=='Sexista' else 'white'
    return f'background-color: {color}'


st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

#st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
#colT1,colT2 = st.columns([2,8])
st.markdown(""" <style> .fondo {
background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Flasmujereseneldeportemexicano.wordpress.com%2F2016%2F11%2F17%2Fpor-que-es-importante-hablar-de-genero%2F&psig=AOvVaw0xG7SVXtJoEpwt-fF5Kykt&ust=1676431557056000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCJiu-a6IlP0CFQAAAAAdAAAAABAJ");
background-size: 180%;} 
</style> """, unsafe_allow_html=True)


st.markdown(""" <style> .font {
font-size:40px ; font-family: 'Cooper Black'; color: #301E67;} 
</style> """, unsafe_allow_html=True)

#st.markdown('<p class="font"; style="text-align: center;>Análisis de comentarios sexistas en linea</p>', unsafe_allow_html=True)
st.markdown('<p class="font" style="text-align: center;">Detectando el Sexismo en Linea: Un proyecto de Investigación</p>', unsafe_allow_html=True)

    
st.markdown(""" <style> .font1 {
font-size:28px ; font-family: 'Times New Roman'; color: #8d33ff;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font2 {
font-size:18px ; font-family: 'Times New Roman'; color: #5B8FB9;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font2">Este proyecto consiste en una aplicación web que utiliza la biblioteca Tweepy de Python para descargar tweets de Twitter, permitiendo buscar Tweets por usuario y por localidad. Luego, utiliza modelos de lenguaje basados en Transformers para analizar los tweets y detectar comentarios y determinar si son "Sexistas" o "No Sexistas". El proyecto busca identificar y combatir el discurso sexista en línea para promover la igualdad de género y la inclusión.</p>',unsafe_allow_html=True)


def tweets_usuario(usuario, cant_de_tweets):
  tabla = []
  if(cant_de_tweets > 0 and usuario != "" ):
      try:
          # Buscar la información del perfil de usuario
          user = api.get_user(screen_name=usuario)
          tweets = api.user_timeline(screen_name = usuario,tweet_mode="extended", count= cant_de_tweets)
          result = []
          for tweet in tweets:
              if (tweet.full_text.startswith('RT')):
                  continue
              else:
                  text = tweet.full_text
                  try:
                      language = detect(text)
                      if language == 'es':
                          datos=limpieza_datos(text)
                          if datos == "":
                              continue
                          else:
                              prediction = pipeline_nlp(datos)
                              for predic in prediction:
                                etiqueta = {'Tweets': datos, 'Prediccion': predic['label'], 'Probabilidad': predic['score']}
                                result.append(etiqueta)
                  except:
                      pass   
          df = pd.DataFrame(result)
          if df.empty:
              muestra= st.text("No hay tweets Sexistas a analizar")
              tabla.append(muestra)
          else:
              df.sort_values(by=['Prediccion', 'Probabilidad'], ascending=[False, False], inplace=True)
              df['Prediccion'] = np.where(df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
              df['Probabilidad'] = df['Probabilidad'].apply(lambda x: round(x, 3))
              muestra = st.table(df.reset_index(drop=True).head(50).style.applymap(color_survived, subset=['Prediccion']))
              if len(df) > 10:
                  # Agregar una barra de desplazamiento vertical a la tabla
                  muestra._parent.markdown(f'<style>.dataframe .data {{height: 300px; overflow: scroll}}</style>', unsafe_allow_html=True)
              tabla.append(muestra)
      except Exception as e:
          muestra = st.text(f"La cuenta {usuario} no existe.") 
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
        tweets = api.search_tweets(q="",lang="es",geocode=f"{location.latitude},{location.longitude},{radius}", count = 100, tweet_mode="extended")
        result = []
        for tweet in tweets:
            if (tweet.full_text.startswith('RT')):
                continue
            elif not tweet.full_text.strip():
                continue
            else:
                datos = limpieza_datos(tweet.full_text)
                prediction = pipeline_nlp(datos)
                for predic in prediction:
                    etiqueta = {'Tweets': datos,'Prediccion': predic['label'], 'Probabilidad': predic['score']}
                    result.append(etiqueta)
        df = pd.DataFrame(result)
        if df.empty:
            muestra=st.text("No se encontraron tweets sexistas dentro de la localidad")
            tabla.append(muestra)
        else:
            #tabla.append(muestra)
            df.sort_values(by=['Prediccion', 'Probabilidad'], ascending=[False, False], inplace=True)
            #df.sort_values(by='Probabilidad', ascending=False, inplace=True)
            #df.sort_values(by='Prediccion', ascending=False, inplace=True)
            df['Prediccion'] = np.where(df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
            df['Probabilidad'] = df['Probabilidad'].round(3)
            # Obtener los datos con probabilidad mayor a 0.50
            df = df[df['Probabilidad'] > 0.50]
            # Obtener los 3 primeros datos con mayor probabilidad sexista
            sexista_df = df[df['Prediccion'] == 'Sexista'].head(3)
            
            # Obtener los 3 primeros datos con mayor probabilidad no sexista
            no_sexista_df = df[df['Prediccion'] == 'No Sexista'].head(3)
            
            # Concatenar ambos dataframes
            muestra_df = pd.concat([sexista_df, no_sexista_df], axis=0)
            col1, col2 = st.columns(2)
            with col1:
                muestra = st.table(muestra_df.reset_index(drop=True).head(6).style.applymap(color_survived, subset=['Prediccion']))
            with col2:
                resultado = df['Prediccion'].value_counts()
                def autopct_fun(abs_values):
                    gen = iter(abs_values)
                    return lambda pct: f"{pct:.1f}% ({next(gen)})"
                    
                colores=["#aae977","#EE3555"]
                fig, ax = plt.subplots()
                fig.set_size_inches(2,2)
                plt.pie(resultado,labels=resultado.index,autopct=autopct_fun(resultado),colors=colores,  textprops={'fontsize': 5})
                ax.set_title("Porcentajes por Categorias en la localidad\n"+buscar_localidad.capitalize(), fontsize=5, fontweight="bold")
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
    language = detect(frase)
    if frase == "":
        tabla = st.text("Ingrese una frase")
        #st.text("Ingrese una frase")
    elif language == 'es':
        predictions = pipeline_nlp(frase)
        # convierte las predicciones en una lista de diccionarios
        data = [{'Texto': frase, 'Prediccion': prediction['label'], 'Probabilidad': prediction['score']} for prediction in predictions]
        # crea un DataFrame a partir de la lista de diccionarios
        df = pd.DataFrame(data)
        df['Prediccion'] = np.where( df['Prediccion'] == 'LABEL_1', 'Sexista', 'No Sexista')
        # muestra el DataFrame
        tabla = st.table(df.reset_index(drop=True).head(1).style.applymap(color_survived, subset=['Prediccion']))
    else:
        tabla = st.text("Solo Frase en español")
        
    return tabla

def run():
    #col1, col2 = st.columns(2)
    with st.form("my_form"):
        search_words = st.text_input("Introduzca la frase, el usuario o localidad para analizar y pulse el check correspondiente")
        number_of_tweets = st.number_input('Introduzca número de tweets a analizar del usuario Máximo 50', 0,50,0)
        st.write("Escoja una Opción:")
        termino=st.checkbox('Frase')
        usuario=st.checkbox('Usuario')
        localidad=st.checkbox('Localidad')
        submit_button = st.form_submit_button(label='Analizar')
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
              tweets_usuario(search_words,number_of_tweets)
          elif (localidad):
              tweets_localidad(search_words)
   
run()
