import streamlit as st
import cv2


title = "Analyse de radiographies pulmonaires Covid-19 🫁"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("streamlit_app/assets/intro_banniere.png")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown("""
## Contexte

La **propagation rapide et étendue de la COVID-19** a engendré des **difficultés** considérables
dans les **systèmes de santé**, notamment en ce qui concerne le diagnostique et le **dépistage des
patients** à un rythme soutenu.

Un **défi** majeur dans la lutte contre la propagation de la maladie a été la **disponibilité 
des tests** retardant ainsi le dépistage de nombreux cas suspects.

Pour palier ce manque, certains centres médicaux ont eut **recourt à la radiographie thoracique**
pour le diagnostique et le suivi des patients atteints de COVID-19.

Toutefois, l'analyse des radiographies nécessite la présence d'un professionnel, 
et le nombre de cas à traiter dans une journée de travail et les signes parfois 
faibles de la COVID sur les radiographies peuvent amener des erreurs de diagnostiques.""")
    
    
    st.markdown("## Objectif du projet")
    st.markdown("""Réaliser un ensemble de modèles permettant l'analyse d'une
radiographie thoracique : 

 1. **Segmentation** de la radio : pour détecter et isoler les poumons du reste de la radio
 2. **Classification** : pour aider le radiologue en prédisant un taux de risque COVID.

Le but n'est pas de réaliser un diagnostique à la place du professionnel de santé,
mais d'attirer son attention lorsqu'une probabilité COVID est détectée.

""")
    
 

    st.markdown("## Caractéristiques observables de la COVID")
    
    st.markdown("### Localisation")
    st.markdown("""Les principales observations des effets de la COVID sont plus fréquemment
faites dans la partie **périphérique bilatérale** et **moyennes et inférieure** des poumons.
""")
    col1, col2, _ = st.columns(3)    
    with col1:
        st.image("streamlit_app/assets/intro_localisation-1.png", use_column_width=True, caption="Localisations d'opacités pulmonaires")
    with col2:
        st.image("streamlit_app/assets/intro_localisation-2.png", use_column_width=True, caption="")
    
    
    st.markdown("### infiltrats en verre dépoli (Ground Glass Opacity)")
    tog_GGO = st.toggle('Inverser la radio', value=True)
    
    col1, col2, _ = st.columns(3)    
    with col1:
        imgNormal = cv2.imread('streamlit_app/assets/intro_GGO_Normal-10160.png', cv2.IMREAD_GRAYSCALE)
        imgNormal = cv2.equalizeHist(imgNormal)
        imgNormal = cv2.resize(imgNormal, (300,300))
        if tog_GGO:
            imgNormal = 255 - imgNormal
        st.image(imgNormal, use_column_width=True, caption='Poumons sains')
        
    with col2:
        imgGGO    = cv2.imread('streamlit_app/assets/intro_GGO_Externe-Covid-GGO.png', cv2.IMREAD_GRAYSCALE)
        imgGGO    = cv2.equalizeHist(imgGGO)
        imgGGO    = cv2.resize(imgGGO, (300,300))
        if tog_GGO:
            imgGGO = 255 - imgGGO
        st.image(imgGGO, use_column_width=True, caption='Poumons avec GGO dû à la COVID')
    

    st.markdown("### Opacités pulmonaires liées à la COVID")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('streamlit_app/assets/intro_opactite_day-1.png', use_column_width=True, caption='Patient A - Jour 1')
    with col2:
        st.image('streamlit_app/assets/intro_opactite_day-3.png', use_column_width=True, caption='Patient A - Jour 3')
    with col3:
        st.image('streamlit_app/assets/intro_opactite_day-11.png', use_column_width=True, caption='Patient A - Jour 11')

    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)
    
    
    
    

    
