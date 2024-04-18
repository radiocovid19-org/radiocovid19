import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import cv2
#from PIL import Image
import matplotlib.pyplot as plt
import time

from sklearn.metrics import classification_report

im_exemple  = 'COVID-1002.png'

title = "Modèle de classification"
sidebar_name = "Classification"

facecolor = '#E8F5F8'
edgecolor = '#50B4C8'


def run():

    st.title(title)
    st.markdown(
        """

        ## Objectif

        Dans cette étape, nous souhaitons réaliser un modèle de **classification binaire** de radios thoraciques permettant l'aide à la détection de cas de COVID-19 (binaire = Non-Covid/Covid).

        ## Echantillon utilisé

        - Le Dataset est celui présenté dans la partie [Data Visualisation]
        - **10.000 images** seront sélectionnées parmi les 21.165 images disponibles
        - **Equilibrage de l'échantillon** de données pour avoir environ 50% de radiographies COVID et 50% autres cas
        """
        )

       
    # Affichage d'un graphique avant/après pour montrer le rééquilibrage
    # Chargement des données 'avant rééquilibrage'
    df_all = pd.read_csv('./streamlit_app/assets/metadata.csv')
    df_tt  = pd.read_csv('./streamlit_app/assets/df_classification_echantillon.csv')

    counts_all = df_all['target'].value_counts(normalize=True).loc[['Normal', 'Lung Opacity', 'Viral Pneumonia', 'COVID']]
    counts_ech = df_tt['target'].value_counts(normalize=True).loc[['Normal', 'Lung Opacity', 'Viral Pneumonia', 'COVID']]
    colors = {'Normal'         : '#3366CC',
              'Lung Opacity'   : '#00B5F7',
              'Viral Pneumonia': '#FF9DA6',
              'COVID'          : '#FB0D0D'}

    bottoms = np.zeros(4)
    
    fig = plt.figure(figsize=(8,5))
    
    for i in ['Viral Pneumonia', 'Lung Opacity', 'Normal', 'COVID']:
    
        bar = plt.bar(x=['Dataset complet\n21.165 images', 'Echantillonnage', 'Echantillon\n10.000 images', 'Résultat'],
                      height = [counts_all[i], 0, counts_ech[i], 0],
                      bottom = bottoms,
                      color  = colors[i],
                      label  = i
                     )
        
        plt.bar_label(bar,
                      label_type='center',
                      fmt    = lambda x: (str(int(round(100*x, 0)))) + '%' if x!=0 else '')
        
        plt.fill([0.4, 0.4, 1.6, 1.6],
                 [bottoms[0], counts_all[i]+bottoms[0], counts_ech[i]+bottoms[2], bottoms[2]],
                 facecolor=colors[i],
                 alpha = 0.1,
                 zorder=0,
                 )
        
        plt.text (s=i,
                  x=1,
                  y=(counts_all[i]+counts_ech[i])/4+(bottoms[0]+bottoms[2])/2,
                  color=colors[i],
                  horizontalalignment='center'
                 )
    
        plt.fill([2.4, 2.4, 3.6, 3.6],
                 [bottoms[2], counts_ech[i]+bottoms[2], counts_ech[i]+bottoms[2], bottoms[2]],
                 facecolor = colors[i] if i=='COVID' else 'blue',
                 alpha = 0.1,
                 zorder=0,
                 )
        
        if i=='COVID':
            plt.plot([0.4,1.6, 3.6],
                    [bottoms[0], bottoms[2], bottoms[2]],
                     color=colors[i],
                     zorder=0,
                     linestyle=':',
                     linewidth=1,
                     )
                     
            plt.text (s='COVID\n50%',
                      x=3,
                      y=(counts_ech[i] / 2 +bottoms[2]),
                      color=colors[i],
                      horizontalalignment='center'
                     )
            plt.text (s='non-COVID\n50%',
                      x=3,
                      y=bottoms[2] / 2,
                      color='black',
                      horizontalalignment='center'
                     )
            
        
        bottoms += [counts_all[i], 0, counts_ech[i], 0]
    
    # plt.legend(loc='lower center')
    plt.title('Echantillonage du Dataset avec équilibrage COVID/non-COVID', loc='left', fontsize='medium')
    plt.yticks([])
    
    st.pyplot(fig)
    exp_augmentation = st.expander("Afficher la décision")
    with exp_augmentation:
        st.markdown(
            """
            L'échantillon est constitué de 10.000 images dont 50% COVID. Nous avons donc **besoin de 
5.000 images COVID** pour avoir un échantillon équilibré. Or le dataset n'en **contient que 3.616**. 
Nous avons donc utilisé plusieurs fois la même image COVID que nous allons modifier en ajoutant au modèle
un bloc de data augmentation.""")
        st.success("Ajout d'un bloc d'Augmentation au modèle.", icon="✅")
    

    ##########################################################################
    # Preprocessing
    ##########################################################################
    st.markdown("## Preprocessing")

    # Masquage ###############################################################
    st.markdown("### Preprocessing : Masquage")
    st.markdown("#### Application du masque sur une radio")

    tog_masquage = st.toggle("Segmenter l'image", value=False)
    col1, col2 = st.columns(2)

    if tog_masquage == False:
        with col1:
            st.image('./streamlit_app/assets/classification_masquage_2.png',
                     caption='Radio thoracique annotée par un professionnel')
        with col2:
            st.markdown("""Comme le montre l'image ci-contre, la COVID a des effets visibles sur 
la partie inférieure et périphérique des poumons.

Les masques fournis dans le dataset devraient a priori inclure ces zones.""")
    else:
        with col1:
            st.image('./streamlit_app/assets/classification_masquage_2_mask.png',
                     caption='Radio thoracique segmentée')
        with col2:
            st.markdown("""Les masques du dataset excluent de l'image une partie des poumons visiblement touchée par la COVID.

Nous supposons que cela peut avoir un impact sur les résultats de notre modèle de classification.
""")

    exp_decision = st.expander("A quoi correspond la région omise ?")
    with exp_decision:
        st.write("""<div class="sketchfab-embed-wrapper"> <iframe title="Adult heart and lung (Animated)" frameborder="0" allowfullscreen mozallowfullscreen="false" webkitallowfullscreen="false" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share width="670" height="500" src="https://sketchfab.com/models/1e8e022d36084490b9e17b55182b9ce7/embed?autostart=1&camera=0&dnt=1"> </iframe> </div>""", unsafe_allow_html=True)

        st.markdown("""La région non comprise dans le masque correspond la partie des poumons en contact avec le diaphragme et la 
région cardiaque. Or les effets de la COVID sont régulièrement observés dans la région basse des poumons.""")

        st.success("Nous décidons d'inclure cette région dans notre segmentation d'image et de mesurer l'impact de cette décision.",
                   icon="✅")

    st.markdown("#### Comment conserver cette zone dans notre masque ?")
    
    mask_dataset, mask_sans, mask_special = st.tabs(["Radio masquée",
                                                     "Radio non masquée",
                                                     "Radio découpée"])
    with mask_dataset:
        col1, col2 = st.columns(2)
        with col1:
            st.image('./streamlit_app/assets/classification_masquages_avec.png', caption='')
        with col2:
            st.markdown("""Avec le masque du **dataset**, la région pulmonaire qui nous intéresse est perdue.\n\n
Nous devons trouver un moyen de la conserver.""")

    with mask_sans:
        col1, col2 = st.columns(2)
        with col1:
            st.image('./streamlit_app/assets/classification_masquages_sans.png', caption='')
        with col2:
            st.markdown("Une première possibilité consiste à conserver l'intégralité de l'image.")
            st.success("Les zones qui nous intéressent sont conservées.", icon='👍')
            st.error("Le CNN aura accès à des zones extérieures à la région pulmonaire.", icon='👎')

    with mask_special:
        col1, col2 = st.columns(2)
        with col1:
            st.image('./streamlit_app/assets/classification_masquages_special.png', caption='')
        with col2:
            st.markdown("""Une seconde possibilité est d'utiliser le masque de segmentation pour 
découper l'image et conserver les parties basses et cardiaque des poumons.""")
            st.success("""Les zones en dehors de la région pulmonaire sont exclues.  
Les régions qui nous intéressent sont conservées.""", icon='👍')
            st.error("La trachée est également conservée.", icon='👎')

    
    st.markdown("#### Comment vérifier l'impact du masquage ?")
    st.markdown("""Nous décidons d'entraîner plusieurs modèles et de comparer leurs performances
en utilisant 3 modes de masquages :""")
    st.success("""
 1. sans masque
 2. avec masque du dataset
 3. avec une radio redécoupée pour conserver la région centrale""", icon='✅')

    # Recadrage ##############################################################
    st.markdown("### Preprocessing : Recadrage")
    st.markdown("""Après application du masque, il apparait :
 - beaucoup de place semble perdue sur nos images
 - la position des poumons n'est pas stable et cela pourrait ralentir l'apprentissage de notre CNN.""")
 
    st.markdown("""Nous décidons d'harmoniser les radios en recadrant les images sur la région
pulmonaire avant redimensionnement en 256x256 pixels.""")

    tog_cadrage = st.toggle("Recadrer les radios", value=False)
    if tog_cadrage == False:
        st.image('./streamlit_app/assets/classification_recadrage_sans.png')
    else:
        st.image('./streamlit_app/assets/classification_recadrage_avec.png')

    st.markdown("#### Vérifier l'impact du recadrage.")
    st.success("""Nous décidons d'entraîner plusieurs modèles (avec ou sans recadrage) et de comparer leurs performances.""", icon='✅')

    # Egalisation ############################################################
    st.markdown("### Preprocessing : Egalisation")
    st.markdown("""Comme vu lors de l'exploration des données, le contraste et la luminosité des radios ne sont 
pas uniformes au sein du dataset. Nous allons donc égaliser les images pour améliorer leur qualité et utiliser 
tout le spectre du niveau de gris.  
Nous avons choisi une **égalisation de type CLAHE**.""")
    exp_exemple = st.expander('Afficher un exemple')
    with exp_exemple:
        st.image('./streamlit_app/assets/classification_egalisation.png')
        st.success("""Après avoir effectué différents tests sur des CNN et compte tenu des résultats articles consultés,
nous avons sélectionné une égalisation CLAHE en découpant l'image en tuiles de 8x8.""", icon='✅')

    # Normalisation ##########################################################
    st.markdown("### Preprocessing : Normalisation")
    st.markdown("""Les images seront normalisées avec les fonctions spécifiques aux modèles préentraînés utilisés.
(par exemple pour VGG16 : `tf.keras.applications.vgg16.preprocess_input()`)

En fonction des modèles, les couches des images pourront être réorganisées, les valeurs des couleurs recentrées 
autour de zéro ou encore remises à l'échelle [-1,+1])""") 


    # Schéma de préprocessing ################################################
    st.markdown("### Préprocessing : Synthèse")
    st.image('./streamlit_app/assets/classification_preprocessing.png')
    
    
    ##########################################################################
    # Modèle de classification
    ##########################################################################
    st.markdown("""## Le modèle de classification

Création d'un modèle de classification **binaire** : COVID/non COVID.

Utililsation de la technique du **transfer learning** : réutilisation d'un modèle pré-entraîné pour résoudre
un nouveau problème  connexe.""")
    
    # Architercture du modèle ################################################
    st.markdown("### Architecture du modèle")    
    fig = plt.figure(figsize=(9,3), layout='tight', facecolor=facecolor, edgecolor=edgecolor, linewidth=1)
    plt.imshow(plt.imread('./streamlit_app/assets/classification_modele.png'))
    plt.axis('off')
    plt.title('Structure du modèle de classification utilisé',
              fontdict={'fontsize':'small'},
              loc='left',
              y = -0.15)
    st.pyplot(fig)

    mod_bloc1, mod_bloc2, mod_bloc3 = st.tabs(["Bloc 1 : couches d'augmentations",
                                               "Bloc 2 : modèle préentraîné",
                                               "Bloc 3 : couches de classification"])

    with mod_bloc1:
        st.markdown("""
 - Rotation des images : [-5%, 5%]
 - Zoom : [-5%, 0]
 - Déplacement vertical : [-12.5%, 12.5%]""")

    with mod_bloc2:
        st.markdown("""3 modèles pré-entraînés ont été testés séparément :
 - VGG16
 - VGG19
 - Xception

Utilisation du **Finetuning** : les différentes couches des modèles pré-entraînés ont été bloquées sauf quelques couches finales.""")
        
        exp_finetuning = st.expander('Afficher le détail du finetuning')
        with exp_finetuning:
            st.write("""
            <table class="myTableStyle" style="width:100%">
            <caption>techniques de Finetuning utilisés pour les différents modèles</caption>
            <thead>
                <tr style="">
                    <th>Modèle préentraîné utilisé</th>
                    <th>Finetuning appliqué</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th>VGG16</th>
                    <td>
                    Entraînement des couches du modèle bloqué, sauf :<br>
                    <ul>
                        <li>dernière couche de Convolution</li>
                        <li>dernière couche de MaxPooling</li>
                    </ul>
                    </td>
                </tr>
                <tr>
                    <th>VGG19</th>
                    <td>
                    Entraînement des couches du modèle bloqué, sauf :<br>
                    <ul>
                        <li>dernière couche de Convolution</li>
                        <li>dernière couche de MaxPooling</li>
                    </ul>
                    </td>
                </tr>
                <tr>
                    <th>Xception</th>
                    <td>
                    Entraînement des couches du modèle bloqué, sauf :<br>
                    <ul>
                        <li>dernier bloc de convolution (block_14, soit 6 couches)</li>
                    </ul>
                    </td>
                </tr>
            </tbody>
            </table>
            <p></p>""", unsafe_allow_html=True)

    with mod_bloc3:    
        st.markdown("""
 - GlobalAveragePooling2D
 - BatchNormalization
 - Dropout(0.5)
 - Dense(512, activation='relu')
 - BatchNormalization
 - Dropout(0.5)
 - Dense(activation='softmax')
 """)
    
    # Paramètres #############################################################    
    st.markdown("""### Paramètres de compilation
 - Optimiseur : Adam
 - Fonction de perte : entropie croisée catégorielle (sparse categorical cross entropy)
 - Métrique: accuracy
 """)

    # Entraînement ###########################################################
    st.markdown("""### Entraînement
 - Epoques : 20
 - Fonctions de callback : `EarlyStopping` et `ReduceLROnPlateau`
""")

    btn = st.button("Entraîner un modèle") 
    
    if btn:
        progress_text = "Entraînement en cours. Veuillez patienter..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in np.linspace(0, 100, 10, endpoint=True, dtype='int'):
            time.sleep(0.10)
            my_bar.progress(int(percent_complete), text=progress_text)
        time.sleep(1)
        my_bar.empty()
        
        st.markdown ("Entraînement d'un modèle de utilisant du transfert learning à partir d'un modèle VGG16 préentraîné.")
        st.image('./streamlit_app/assets/classification_entrainement.png')
        time.sleep(0.5)
        st.success("""Au bout de **8** époques, l'accuracy sur le jeu de validation est de **92%** 
avec une perte 'sparse categorical cross entropy' de **0.22**.""", icon='✅')

    
    # Performances des modèles ###############################################
    st.markdown("### Résultats")
    st.markdown("""18 modèles entraînés.  
En fonction des modèles, l'accuracy varie de **0.81** à **0.97**.""")

    exp_resultats = st.expander('Afficher le tableau complet')
    with exp_resultats:
        df = pd.read_pickle('./streamlit_app/assets/classification_resultats.pickle')
        df.index = df.index.droplevel(1)
        st.dataframe(df.style.format(precision=2), use_container_width=True)
    


    ###########################################################################
    # Interprétation des résultats
    ###########################################################################
    st.markdown("## Interprétation des résultats")
    
    # Masquage ################################################################
    st.markdown("""### Impact du masquage""")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.pivot_table(df, values='Accuracy', index='Masquage', columns='Modèle')
                     .sort_values('VGG16', ascending=False)
                     .style.format(precision=2),
                    use_container_width = True)
    with col2:
        st.markdown("Les résultats sans l'utilisation des masques sont nettement meilleurs (+10 pts en moyenne).")

    
    st.markdown("""Pour expliquer pourquoi les résultats varient autant, et comprendre ce qu'utilisent 
    les CNN pour effectuer leur classification, nous allons utiliser la technique du **GradCAM**.""")
    
    mask_sans, mask_poumons, mask_special, mask_dataset = st.tabs(["Sans masque", "Poumons cachés", "Masque Spécifique", "Masque du dataset"])
    with mask_sans:
        tog_gradcam_ff = st.toggle('Activer GradCAM', value=False, key='t_ff')
        if tog_gradcam_ff:
            img  = './streamlit_app/assets/classification_gradcam_false_false.png'
            txt  = "**Prédiction COVID = 100%**.\n\n"
            txt += "Pour effectuer sa classification, le CNN a utilisé des informations extérieures à la zone pulmonaire.\n\n"
            txt += "Ces modèles donnent de bons résultats, mais ils sont en réalité biaisés.\n\n"
            txt += "**La segmentation et le masquage de l'image semblent indispensables**."
        else:
            img = './streamlit_app/assets/classification_gradcam_image_false_false.png'
            txt = '**Prédiction COVID = 100%**'
        col1, col2 = st.columns(2)
        with col1:
            st.image(img)
        with col2:
            st.markdown(txt)
            
    with mask_poumons:
        tog_gradcam_pm = st.toggle('Activer GradCAM', value=True, key='t_pm')
        if tog_gradcam_pm:
            img  = './streamlit_app/assets/classification_gradcam_rectangle-noir.png'
            txt  = "**Prédiction COVID = 100%**\n\n"
            txt += "En effet, le modèle utilise l'annotation sur la radio."
        else:
            img = './streamlit_app/assets/classification_gradcam_image_rectangle-noir.png'
            txt = '**Prédiction COVID = 100%**\n\n'
            txt+= "Avec ce test effectué en utilisant le même modèle que précédemment, on obtient le score étonnant de **100%**. \n\n"
            txt+= "Cela confirme que le modèle n'utilise pas la région pulmonaire pour effectuer sa classficiation"
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(img)
        with col2:
            st.markdown(txt)

    with mask_special:
        tog_gradcam_sf = st.toggle('Activer GradCAM', value=True, key='t_sf')
        if tog_gradcam_sf:
            img  = './streamlit_app/assets/classification_gradcam_special_false.png'
            txt  = '**Prédiction COVID = 95%**\n\n'
            txt += "Ici, le réseau de neurones est contraint de respecter la région pulmonaire.\n\n"
            txt += "La prédiction est réalisée en utilisant des zones périphériques et basses des poumons."
        else:
            img = './streamlit_app/assets/classification_gradcam_image_special_false.png'
            txt = '**Prédiction COVID = 95%**'
        col1, col2 = st.columns(2,)
        with col1:
            st.image(img)
        with col2:
            st.markdown(txt)

    with mask_dataset:
        tog_gradcam_tf = st.toggle('Activer GradCAM', value=True, key='t_tf')
        if tog_gradcam_tf:
            img  = './streamlit_app/assets/classification_gradcam_true_false.png'
            txt  = "**Prédiction COVID = 65%**\n\n"
            txt += "En activant la gradCAM, on constate que le CNN ne peut plus utiliser la partie basse centrale des poumons "
            txt += "comme il le faisait avec le masque spécifique."
        else:
            img  = './streamlit_app/assets/classification_gradcam_image_true_false.png'
            txt  = "**Prédiction COVID = 65%**\n\n"
            txt += "Le taux prédit pour la classe COVID a chuté : \n"
            txt += "95%->65%."
            
        col1, col2 = st.columns(2,)
        with col1:
            st.image(img)
        with col2:
            st.markdown(txt)

    exp_mask = st.expander('Afficher la conclusion')
    with exp_mask:
        st.success("""Nous préconisons **l'utilisation d'un masque qui conserve les parties basses des poumons**.
C'est d'ailleurs dans cette région, en plus de la périphérie des poumons, 
que des effets de la COVID sont souvent constatés.""", icon='✅')

    # Recadrage ###############################################################
    st.markdown("""### Impact du recadrage""")
    col1, col2 = st.columns(2,)
    with col1:
        st.dataframe(pd.pivot_table(df[df['Masquage']=='Spécifique'], values='Accuracy', index='Recadrage', columns='Modèle')
                     .sort_values('VGG16', ascending=True)
                     .style.format(precision=2),
                    use_container_width = True)
    with col2:
        st.markdown("""Le recadrage semble avoir un impact positif sur la précision (+1 point en moyenne)""")

    exp_mask = st.expander('Afficher la conclusion')
    with exp_mask:
        st.success("""Nous préconisons **un recadrage de l'image** sur la régions masquée.""", icon='✅')
    

    ###########################################################################
    # Sélection du meilleur modèle
    ###########################################################################
    st.markdown("## Sélection du meilleur modèle")
    st.dataframe(df[(df['Masquage']=='Spécifique')&(df['Recadrage']=='Oui')]
                 .sort_values('Accuracy', ascending=False)
                 .style.format(precision=2),
                use_container_width = True)
    st.markdown("Le modèle qui donne les meilleurs résultats est le modèle utilisant un modèle préentraîné VGG16.")

    
    st.markdown("### Définition du seuil non-COVID / COVID")
    st.image('streamlit_app/assets/classification_performance-ROC.png')
    st.markdown("La valeur de seuil qu permet une séparation optimale des cas non COVID des cas COVID est : **0.377**.")
    st.success("""C'est à partir de ce seuil que le voyant d'alerte COVID passera de vert au rouge 
dans notre application **RadioCovid19**.""", icon='✅')


    st.markdown("### Matrices de confusion")
    
    probas = pd.read_csv('streamlit_app/assets/classification_tableau-proba.csv')

    tog_normalise = st.toggle('Normaliser', value=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**Globale**')
        ct = pd.crosstab(probas['true'].replace({0:'Non-covid', 1:'COVID'}),
                         probas['pred'].replace({0:'Non-covid', 1:'COVID'}),
                         rownames=['classes réelles :'],
                         colnames=['classes prédites'],
                         normalize='index' if tog_normalise else False)
        # Mise en forme
        ct_styled = ct.style
        ct_styled.format(formatter='{:.0%}' if tog_normalise else None,
                         precision=2)
        ct_styled.apply(styler, axis=0)
        st.write(ct_styled.to_html(), unsafe_allow_html=True)

    with col2:
        st.markdown('**Par pathologie**')
        # Calcule la matrice de confusion
        ct = pd.crosstab(probas['type'],
                         probas['pred'].replace({0:'Non-covid', 1:'COVID'}),
                         rownames=['Type pathologie :'],
                         colnames=['classes prédites'],
                         normalize='index' if tog_normalise else False)
        # Mise en forme
        ct_styled = ct.style
        ct_styled.format(formatter='{:.0%}' if tog_normalise else None,
                         precision=2)
        ct_styled.apply(styler, axis=0)
        st.write(ct_styled.to_html(), unsafe_allow_html=True)

    st.markdown("<div style='padding-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    exp_matrices = st.expander("Afficher l'interprétation")
    with exp_matrices:
        st.markdown("""La matrice de confusion globale Covid/non-Covid nous montre que la classification est bien gérée par le modèle.
Les résultats sont équilibrés.

La matrice de confusion détaillée par pathologie indique que les radios "Viral Pneumonia" sont les moins bien prédites. Cette classe étant
minoritaire, il y a peu de cas traités par le modèle pendant la phase d'entraînement (274 radios sur 8000 du jeu d'entraînement). 
Pour y remédier, nous aurions pu augmenter le nombre de radios "Viral Pneumonia" de notre jeu d'entraînement et détriment des autres pathologies.
""")
    
    
    st.markdown("### Rapport de classification")
    cr = classification_report(probas['true'], probas['pred'], target_names=['non-Covid','Covid'])
    st.markdown(f"""<span></span>
```
.{cr}
```
""", unsafe_allow_html=True)
    

    exp_classreport = st.expander("Afficher l'interprétation")
    with exp_classreport:
        st.markdown("""La **précision** (capacité à bien classer les cas) est bonne, et le **rappel** (taux de bonne prédiction) l'est également.  

Le modèle a un taux de bonne prédiction satisfaisant (**accuracy**) de 92%.
""")
        st.success("ce modèle sera donc celui que nous utiliserons dans notre application finale **RadioCovid19**.", icon='✅')

    
    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)




def styler(serie):
    """ Mise en forme des matrices de confusion
    """
    style = []
    col_name = serie.name
    for (index,value) in serie.to_dict().items():
        if index in ('Lung_Opacity', 'Normal', 'Viral Pneumonia'):
            index = 'Non-covid'
        if index==serie.name:
            style.append('background-color:#e6ffe6; font-weight: bold;')
        else:            
            style.append('background-color:#ffe6e6;')
    return style
    
    
    

    
    

        

            
            
