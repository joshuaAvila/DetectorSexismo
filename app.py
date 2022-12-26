
import streamlit as st


colT1,colT2 = st.columns([2,8])
with colT2:
   # st.title('Analisis de comentarios sexistas en Twitter') 
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

      
  with st.form("my_form"):
   st.write("Inside the form")
   slider_val = st.slider("Form slider")
   checkbox_val = st.checkbox("Form checkbox")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")    
      
