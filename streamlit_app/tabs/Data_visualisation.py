import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageFilter, ImageStat

import matplotlib.pyplot  as plt
from matplotlib import style
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


title = "DATA"
sidebar_name = "Data Visualisation"
#______________________________________________________________________________________________________
#chargement du dataframe

#/home/souma/juin23_cds_radio_covid-main/streamlit_app/tabs/../../data/metadata.csv

df=pd.read_csv("streamlit_app/assets/metadata.csv",index_col=0)


#chargment d'un exemple d'image et son masque

exemple_image = Image.open('data/COVID/images/COVID-13.png')

exemple_mask = Image.open('data/COVID/masks/COVID-13.png')


#chargement des images pour étudier la luminosité
img_sombre  = Image.open('data/COVID/images/COVID-17.png')
img_moyenne = Image.open('data/COVID/images/COVID-232.png')
img_claire  = Image.open('data/COVID/images/COVID-195.png')

data_sombre = np.array(img_sombre.getdata())
data_moyenne = np.array(img_moyenne.getdata())
data_claire = np.array(img_claire.getdata())

#étude de la variable explicative URL
df["SOURCE"]=df["URL"].replace({'https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data': "Kaggle.com",
                                'https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia':"Kaggle.com",
                                'https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711':"bimcv.cipf.es",
                                'https://github.com/armiro/COVID-CXNet':"github.com",
                                'https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png':"github.com",
                                "https://github.com/ieee8023/covid-chestxray-dataset":"github.com",
                                "https://eurorad.org":"eurorad.org",
                                "https://sirm.org/category/senza-categoria/covid-19/":"sirm.org"})


#______________________________________________________________________________________________________________________________

def run():
   
    st.title(title)
    data=st.toggle('Afficher un extrait du DataFrame', value=True)
    if data:
        st.dataframe(df.head(5))
    
    on = st.toggle('Afficher les dimensions du DataFrame', value=True)

    if on:
        st.write("Ce DataFrame contient",df.shape[0],"lignes et",df.shape[1],"colonnes")

   #########################################################################################################################""

    st.write("**Nous allons maintenant analyser chaque variable du DataFrame**")

    tab1, tab2, tab3 = st.tabs(["Variable cible", "Variable explicative FORMAT & SIZE","Variable explicative Source"])

    with tab1:
       st.header("Etude de la répartition des données")
    
    
       fig_1 = go.Figure(data=[go.Pie(labels=[ "Normal", "Lung_Opacity","Covid","Viral Pneumonia"], 
                                      values=df["target"].value_counts(),
                                      pull=[0, 0, 0.2, 0],
                                      marker_colors = ['#3366CC','#00B5F7','#FB0D0D','#FF9DA6'])])
       st.plotly_chart(fig_1, height=1000, width=700)


       expander_1 = st.expander("***Voir l'interprétation***")
       expander_1.write("""On observe un déséquilibre marqué dans la répartition des données, avec la classe "Normal" qui constitue la
                        majorité, représentant près de la moitié de l'ensemble des données, soit 48%. En revanche, les classes "Covid"
                        et "Viral Pneumonia" sont nettement minoritaires, affichant respectivement des pourcentages de 17,08% et
                        6,35%. Enfin pour la classe “Lung_Opacity” elle relativement importante avec un pourcentage de 28.41%.""")

    with tab2:
       st.header("Etude des formats et des tailles des images")
       st.write("""Les images sont toutes au même formats 'PNG' et de même tailles '299*299' ,ce qui signifie qu’a priori il ne sera pas nécessaire de procéder
                       à un redimensionnement ou à une conversion.""")

    with tab3:
       st.header("Etude des sources des images")

       fig_4=px.histogram(df,x='SOURCE', color='target',color_discrete_sequence=['#FB0D0D','#3366CC','#FF9DA6','#00B5F7'])
       st.plotly_chart(fig_4)


       expander_2 = st.expander("***Voir l'interprétation***")
       expander_2.write("""Il est notable que la majorité des données proviennent du site "Kaggle.com", en particulier celles liées aux
                           classes "Normal", "Viral Pneumonia" et "Lung Opacity". En revanche, les données de la classe "Covid" ont été
                           obtenues à partir de différents sites autres que Kaggle.com (github,sirm.org,eurorad.org,bimcv.cipf.es).""")
       
        ###############################################################################################################
    
    st.title("Aperçu d'une image et son masque")

    col1, col2 = st.columns(2)

    with col1:
           
           st.image("streamlit_app/assets/exemple_image.png", caption='Image COVID-13')
           st.write(f"Taille : {exemple_image.size} pixels")
           st.write(f"Format : {exemple_image.format}")
           st.write(f"Mode : {exemple_image.mode} (L=8 bits pixels, grayscale)")

    with col2:
            
            st.image("streamlit_app/assets/exemple_mask.png", caption='Masque COVID-13')
            st.write(f"Taille : {exemple_mask.size} pixels")
            st.write(f"Format : {exemple_mask.format}")
            st.write(f"Mode : {exemple_mask.mode} (RGB=3x8-bit pixels, true color)")
    
   


    ################################################################################################################

    st.title("Luminosité des radios")


    col3, col4,col5 = st.columns(3)

    with col3:
           
           st.image("streamlit_app/assets/image_sombre.png", caption='Image sombre')
           

    with col4:
            
            st.image("streamlit_app/assets/image_moyenne.png", caption='Image moyenne')
           

    with col5:
            
            st.image("streamlit_app/assets/image_claire.png", caption='Image claire')
          

    fig = go.Figure()

    # Ajouter les histogrammes
    fig.add_trace(go.Histogram(x=data_sombre, nbinsx=64, name='Sombre'))
    fig.add_trace(go.Histogram(x=data_moyenne, nbinsx=64, name='Moyenne'))
    fig.add_trace(go.Histogram(x=data_claire, nbinsx=64, name='Claire'))

    # Calculer les médianes
    m_1 = np.median(data_sombre)
    m_2 = np.median(data_moyenne)
    m_3 = np.median(data_claire)

    fig.add_vline(m_1)
    fig.add_vline(m_2)
    fig.add_vline(m_3)

    fig.add_annotation(x=102, y=4500, text=m_1, bgcolor='black')
    fig.add_annotation(x=132, y=4500, text=m_2, bgcolor='black')
    fig.add_annotation(x=197, y=4500, text=m_3, bgcolor='black')


    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)

    # Afficher le graphique
    st.plotly_chart(fig)

    expander_3 = st.expander("***Voir l'interprétation***")
    expander_3.write("""On peut constater que pour la première image (image sombre), la majorité des pixels se situent dans la partie
            gauche de l'histogramme, vers les valeurs de niveaux de luminosité faibles. Inversement, les pixels de la
            troisième image (image claire) se concentrent dans la partie droite de l'histogramme vers les valeurs de niveaux
            de luminosité fortes. En revanche, pour la deuxième image (image moyenne), il y a un équilibre entre les
            valeurs de pixel d’où une image avec une bonne luminosité.D'où la nécessité de réaliser une égalisation des histogrammes.""")
    
    #################################################################################################################################
    
    st.title("Contraste des radios")
    st.image("streamlit_app/assets/contraste.png")
    expander_4 = st.expander("***Voir l'interprétation***")
    expander_4.write("""Sur l'image à contraste faible, nous voyons sur l'histogramme de gauche que le spectre des couleurs est resserré autour de
                                la médiane. A contrario, sur l'histogramme de droite, le spectre est beaucoup plus étendu. Ce qui tend à montrer que
                                l'image de droite aura beaucoup plus de valeurs de couleurs que l'image de gauche, et donc plus de contraste.
                                Un écart type plus élevé indique donc une plus grande variabilité des niveaux de gris, tandis qu'un écart type plus
                                faible signifie une plus faible variabilité.""")


    ##############################################################################################################################""
   
    st.title("Photos perturbateurs")

    st.write("""Une multitude de matériels (cathéters, tuyaux d'oxygène, sondes électrocardiographiques, pontages aorto-
            coronariens...) sont fréquemment visibles sur les radios thoraciques.
            Certaines radios comportent également des annotations sur la région des poumons (flèches, écriture, zones
            entourées...)
            Leur présence sur les radios risque éventuellement perturber l'analyse des radios.""")
    st.image("streamlit_app/assets/images_perturbateurs.png",caption="Perturbateurs d'apprentissage")

    

    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)