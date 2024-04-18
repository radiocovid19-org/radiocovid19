import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageFilter, ImageStat
import time

import matplotlib.pyplot  as plt
from matplotlib import style
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


title = "Modélisation pour prédiction des masques"
sidebar_name = "Modélisation pour prédiction des masques"

#___________________________________________________________________________________________________________________________________
#chargement du dataframe

df=pd.read_csv("data/metadata.csv",index_col=0)

#____________________________________________________________________________________________________________________________________

def run():

    st.title(title)
    st.header("Objectif")
    st.write("""Prédire la région pulmonaire sur les nouvelles radios thoraciques en vue de la classification binaire COVID/Non-COVID.""")
    st.header("Jeu de données")
    st.write("""Le jeu de données étant volumineux,nous faisons le choix de le limiter à un échantillon constitué d'un nombre optimum d'images.
            """)
    st.markdown(
    "- **6000** images seront sélectionnées parmi les 21 165 images disponibles\n"
    "- **Équilibrage de l'échantillon** de données pour avoir environ 50% de radiographies COVID et 50% autres cas"
    )
    if st.button("Justification du choix de la taille de l'échantillon"):
        st.image("streamlit_app/assets/segmentation_nb_images.png", caption="Recherche de la taille optimum du jeu d'entraînement")
        st.success("Le nombre d'échantillon retenu est 6000 ", icon="✅")

    if st.button("Afficher la nouvelle répartition des données"):
       
       col1, col2 = st.columns(2)

       with col1:
                
                fig_1 = go.Figure(data=[go.Pie(labels=[ "Normal", "Lung_Opacity","Covid","Viral Pneumonia"], 
                                            values=df["target"].value_counts(),
                                            pull=[0, 0, 0.2, 0],
                                            marker_colors = ['#3366CC','#00B5F7','#FB0D0D','#FF9DA6'])])
                st.plotly_chart(fig_1, use_container_width=True)



       with col2:
                    
                    fig_2 = go.Figure(data=[go.Pie(labels=[ "Covid", "Autres"], 
                                            values=[50,50],
                                            pull=[0, 0, 0.2, 0],
                                            marker_colors = ['#FB0D0D','#00B5F7'])])
                    st.plotly_chart(fig_2, use_container_width=True)
    
    


    st.image("streamlit_app/assets/segmentation_preprocessing.png", caption="Les étapes de préprocessing")
    
    st.header("Modélisation")
    st.markdown("### Architecture U-Net pour la segmentation des images")
    st.image("streamlit_app/assets/segementation_model.png", caption='Modèle U-Net construit pour la segmentation des radios')
    


    st.markdown("### Paramètres de compilation 🔧")

    st.markdown(

    "- **Optimiseur**: Adam\n"
    "- **Learning rate**: 0.001\n"
    "- **Fontion de perte**: binary_crossentropy\n"
    "- **Métrique**: Binary Intersection-Over-Union"
     )
    


    st.header("Analyse de la performance du modèle")
    if st.button("Afficher le résultat de l'entrainement"):
       progress_text = "Operation in progress. Please wait."
       my_bar = st.progress(0, text=progress_text)
       for percent_complete in range(100):
          time.sleep(0.01)
          my_bar.progress(percent_complete + 1, text=progress_text)
       time.sleep(1)
       my_bar.empty()
       st.image("streamlit_app/assets/segmentation_performance_du_model.png", caption='Evolution de la métrique “binary_IoU” et de la perte en fonction des époques')
       st.success("""Au bout de ***16*** époques, le Binary IoU sur le jeu de validation est de ***97%*** avec une perte 'binary_crossentropy'
                   de ***0.022*** """, icon='✅')

    
    st.header("Prédiction")
    st.write("""Les résultats de prédictions sont illustrés sur la Figure ci-dessous. La surface de superposition de la
             prédiction (en vert) et le masque réel (en rouge) est présentée en jaune dans la colonne de droite.""")
    on = st.toggle('Afficher la prédiction')

    if on:
        st.image("streamlit_app/assets/segmentation_resultat.png", caption='Comparaison Prédiction vs Masque réel')
        # st.image("streamlit_app/assets/segmentation_resultat_multiple.png", caption='Exemple2:Comparaison Prédiction vs Masque réel')
    

    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)

  
    
