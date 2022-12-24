import streamlit as st
import pandas as pd
import numpy as np
import re
import pysentimiento


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

def run():
    with st.form(key='Introduzca Texto'):
        col,buff1, buff2 = st.columns([2,2,1])
        #col.text_input('smaller text window:')
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
            
           
            if (error == False):
                if (termino):
                    new_search = search_words + " -filter:retweets"
                    tweets =tw.Cursor(api.search_tweets,q=new_search,lang="es",since=date_since).items(number_of_tweets)
                elif (usuario):
                    tweets = api.user_timeline(screen_name = search_words,count=number_of_tweets)
                
                tweet_list = [i.text for i in tweets]
                #tweet_list = [strip_undesired_chars(i.text) for i in tweets]
                text= pd.DataFrame(tweet_list)
                #text[0] = text[0].apply(preprocess)
                text[0] = text[0].apply(preprocess_tweet)
                text1=text[0].values
                indices1=tokenizer.batch_encode_plus(text1.tolist(),
                                         max_length=128,
                                         add_special_tokens=True, 
                                         return_attention_mask=True,
                                         pad_to_max_length=True,
                                         truncation=True)
                input_ids1=indices1["input_ids"]
                attention_masks1=indices1["attention_mask"]
                prediction_inputs1= torch.tensor(input_ids1)
                prediction_masks1 = torch.tensor(attention_masks1)
                # Set the batch size.  
                batch_size = 25
                # Create the DataLoader.
                prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
                prediction_sampler1 = SequentialSampler(prediction_data1)
                prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)
                print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))
                # Put model in evaluation mode
                model.eval()
                # Tracking variables 
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
                flat_predictions = np.argmax(flat_predictions, axis=1).flatten()#p = [i for i in classifier(tweet_list)]
                df = pd.DataFrame(list(zip(tweet_list, flat_predictions)),columns =['Últimos '+ str(number_of_tweets)+' Tweets'+' de '+search_words, 'Sexista'])
                df['Sexista']= np.where(df['Sexista']== 0, 'No Sexista', 'Sexista')
                
                
                st.table(df.reset_index(drop=True).head(20).style.applymap(color_survived, subset=['Sexista']))

                
                #st.dataframe(df.style.apply(highlight_survived, axis=1))
                #st.table(df)
            #st.write(df)
run()
