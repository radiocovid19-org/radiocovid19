import streamlit as st
import pandas as pd
import numpy as np

title = "Conclusion & perspectives"
sidebar_name = "Conclusion & perspectives"


def run():

    st.title(title)

    st.header("Bilan")

    st.markdown(""" 
             - Nous avons réussi à atteindre l'objectif de ce projet et çela en introduisant un modèle d'apprentissage profond
dédié à la détection de la COVID-19 à partir d'images de radiographie thoracique, en optimisant trois modèles de convolution
pré-entraînés (**VGG16**, **VGG19** et **Xception**).
             
- Nous avons fait le choix de sélectionner :blue[**VGG16**] comme meilleur modèle en raison de sa robustesse et 
sa performance par rapport aux autres modèles.

- L'utilisation de la méthode **Grad-CAM** pour interpréter les décisions du modèle a joué un rôle crucial dans la compréhension du fonctionnement du réseau neuronal.

- Le modèle a généré des meilleures performances en préparant le jeu d'entraînement avec un **masquage spécifique** et un **recadrage sur la ROI**.

- L'application construite est **rapide et fiable** ce qui répond à la problèmatique posée au début de projet.""")
    st.header("Pistes d’améliorations")

    st.markdown(""" 
                Certaines limites de notre étude peuvent être surmontées dans des futures recherches à travers:
             
                
             - ***Une amélioration de la qualité*** des images radiographiques en ***éliminant les éléments perturbateurs***
              tels que les annotations sur la région des poumons (flèches, écriture, zones entourées…) et les dispositifs
              médicaux (électrodes, cathéters, tuyaux d'oxygènes…).
                
             - ***Ré-entraînement d'un nouveau modèle*** de segmentation des radios qui permettrait de mieux conserver la partie basse des poumons tout en évitant le redécoupage des images tel que nous l’avons
              fait ici. Cela signifie qu’il faudrait ***repartir d’un nouveau dataset d’images segmentées manuellement*** que nous n’avions pas à notre disposition dans le cadre de cette étude.

            """)
    
    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)











   
   

  
    
