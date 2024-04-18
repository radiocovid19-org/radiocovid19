import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import requests
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from streamlit_image_select import image_select
from streamlit.components.v1 import html
# Imports utiles à la gestion de l'upload d'images
from streamlit.runtime import media_file_manager
from streamlit.runtime import Runtime

title = "RadioCovid19"
sidebar_name = "RadioCovid19"

radios_patients = [# ("streamlit_app/assets/radios/Lung_Opacity-16.png",    'Opacité pulmonaire'),
                   ("streamlit_app/assets/radios/Lung_Opacity-11.png",    'Opacité pulmo.'),
                   ("streamlit_app/assets/radios/Lung_Opacity-286.png",   'Opacité pulmo.'),
                   ("streamlit_app/assets/radios/Normal-27.png",          'Normal'),
                   ("streamlit_app/assets/radios/Normal-70.png",          'Normal'),
                   ("streamlit_app/assets/radios/Normal-8111.png",        'Normal'),
                   ("streamlit_app/assets/radios/Viral Pneumonia-37.png", 'Pneumonie vir.'),
                   ("streamlit_app/assets/radios/Viral Pneumonia-21.png", 'Pneumonie vir.'),
                   ("streamlit_app/assets/radios/COVID-3392.png",         'COVID'),
                   ("streamlit_app/assets/radios/COVID-3081.png",         'COVID'),
                   ("streamlit_app/assets/radios/COVID-1210.png",         'COVID'),
                   ("streamlit_app/assets/radios/COVID-1996.png",         'COVID')]


def run():

    # Init ###################################################################
    st.config.set_option('theme.base' ,"dark")    

    # Chargement des modèles
    @st.cache_resource
    def load_model(url):
        return tf.keras.saving.load_model(url)
    mod_classification = load_model('./streamlit_app/assets/models/classification_VGG16_special_True_clahe')
    mod_segmentation   = load_model('./streamlit_app/assets/models/mask_detection_20240130')

    # Init pour la gestion des images sélectionnées dans la liste ou chargées
    if 'prev_img' not in st.session_state:
        st.session_state.prev_img = None
    if 'uploaded_img' not in st.session_state:
        st.session_state.uploaded_img = None

    # Charge une feuille de style propre à la DEMO
    with open("streamlit_app/tabs/radiocovid19.css", "r") as f:        
        style = f.read()
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

    st.title(title)
    st.text("Ceci est une démonstration d'un projet de Data Science et non un réel outil de diagnostique.")

    col0, col00 = st.columns([4, 1])
    
    with col0:
        selected_img = image_select(label    = "Sélectionnez la radio d'un patient :",
                                    images   = [radios_patients[i][0] for i in range(0, len(radios_patients))],
                                    captions = [radios_patients[i][1] for i in range(0, len(radios_patients))],
                                    use_container_width=False,
                                    key      = 'gallery')

        # Détecte le changement de sélection
        if st.session_state.prev_img != selected_img:
            is_image_changee = True
            st.session_state.prev_img = selected_img
        else:
            is_image_changee = False
    
    with col00:
        uploaded_file = st.file_uploader("Ou chargez une nouvelle image :", type=['png'])
        
        if (uploaded_file is not None):
            # st.write("Le fichier a été téléchargé avec succès")
            is_image_uploadee = True
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_img = cv2.imdecode(file_bytes, 1)
            uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
            st.session_state.uploaded_img = uploaded_img
    
            # Supprime le fichier du gestionnaire de chargement
            session_id = media_file_manager._get_session_id()
            Runtime.instance().uploaded_file_mgr.remove_session_files(session_id=session_id)
            # Nettoie le cache des anciens fichiers qui ont pu rester
            Runtime.instance().media_file_mgr.remove_orphaned_files()
            Runtime.instance().media_file_mgr._get_inactive_file_ids()
        else:
            is_image_uploadee = False

    # Enregistre la dernière image sélectionnée dans le cache de la session
    if is_image_uploadee:
        st.session_state.img = uploaded_img
    elif is_image_changee:
        st.session_state.img = selected_img
    img = st.session_state.img

    st.markdown("---")
    
    # Affichage de la partie basse avec la radio originale, la radio modifiée,
    # et le panneau de commande
    col1, col2, col3 = st.columns(3)

    with col1:
        # Col 1 : image originale
        st.image(img, use_column_width=True)

    with col2:
        # Col 2 : Image Résultat
        # Affiche uniquement un conteneur
        # Le panneau de contrôles étant dans la colonne 3,
        # le contenu de la colonne 2 sera généré plus bas dans ce code.
        cont_col2 = st.container()
        
    with col3:
        # Panneau de commande
        tog_ameliorer = st.toggle("Améliorer la radio")
        tog_zoom      = st.toggle("Zoomer sur la région pulmonaire")
        # tog_masquer   = st.toggle("Mettre en évidence les poumons")
        tog_covid     = st.toggle("Aide au diagnostique")
        sel_gradcam = st.selectbox("Type GradCam :",
                                   ('Encadrer', 'Heatmap', 'Accentuation', 'Sans effet'),
                                   disabled = not(tog_covid),
                                   label_visibility='visible')
        type_gradcam = {'Encadrer':'rectangle', 'Heatmap':'normal', 'Accentuation':'emphasize', 'Sans effet':'normal'}[sel_gradcam]
        interpolation = cv2.INTER_NEAREST if (sel_gradcam == 'Sans effet') else cv2.INTER_CUBIC

        st.markdown("\n  \n  ")

        cont_log = st.container(border=True)

        if tog_zoom:
            mask = mod_segmentation
            cropping=True
        else:
            mask = False
            cropping=False
            
        if tog_ameliorer:
            egalisation = 'CLAHE'
        else:
            egalisation = False

        if (tog_ameliorer or tog_zoom or tog_covid) == False:
            with cont_log:
                # Retourne un texte vide pour afficher le cadre du conteneur
                # Sinon, la cadre est vide, et il ne s'afffiche pas du tout
                st.text (" ") 
                
        else:
            with cont_log:
                # Vide la colonne 2
                with cont_col2:
                    st.empty()

                if is_image_uploadee:
                    st.text("Chargement de l'image : " + uploaded_file.name)
                    
                st.text("[Améliorations]")
                pp_img, log, roi = preprocessing(img, mask,
                                                 masking=False,
                                                 egalisation=egalisation,
                                                 cropping=cropping)
                st.text(log)
                
                if tog_covid:
                    # La reherche COVID est demandée
                    st.text("[Diagnostique]")
                    # Préprocesse l'image avec le préprocessing propre au modèle utilisé dans cette page
                    pp_covid, _, roi = preprocessing(img,
                                                   mod_segmentation,
                                                   masking  = 'special',
                                                   cropping = True,
                                                   egalisation = 'CLAHE',
                                                   normalisation = 'VGG16')
                    st.text("Exécution modèle : RadioCovid19")
                    
                    # Calcule predictions et gradcam
                    heatmap, preds = gradCam(image =pp_covid.reshape(1, 256, 256, 3),
                                             model =mod_classification,
                                             core_model_layername = 'vgg16',
                                             last_conv_layername = 'block5_conv3',
                                             pred_index = 1,
                                             weighted_by_pred = True,
                                             interpolation=interpolation)

                    # Affiche la prédiction COVID
                    pred_covid = round(100 * preds.numpy()[0][1], 1)
                    msg_covid = f"Probabilité COVID : {pred_covid} %"

                    # La limite COVID/non-COVID est fixée à 38% 
                    # conformément aux résultats calculés dnas le § ROC curve
                    if pred_covid <= 38:
                        st.success(msg_covid, icon='🟢')
                    else:
                        st.error(msg_covid, icon='🔴')
                    
                    if tog_zoom == False:
                        # Redimensionne et replace la heatmap
                        # qui a été calculée avec une radio 'croppée' 
                        # et qui doit être placée sur une radio entière
                        heatmap = cv2.resize(heatmap, (roi['x2']-roi['x1'], roi['y2']-roi['y1']))
                        zeros = np.zeros((256,256), dtype='uint8')
                        zeros[roi['y1']:roi['y2'], roi['x1']:roi['x2']] = heatmap
                        heatmap = zeros
                    

            with cont_col2:
                if (tog_covid) and (pred_covid > 50):
                    st.image(superimpose((255*pp_img).astype('uint8'), heatmap, type=type_gradcam),
                             use_column_width=True)
                else:
                    st.image(pp_img, use_column_width=True)
                    

    # Modification de la feuille de style de streamlit-image-select ##########
    script = """
    <script language='javascript'>

    function add_CSS(DOMsis) {
        //
        // Ajoute des styles CSS au code HTML chargé
        //
        
        // Si la feuille de style n'est pas déjà définie
        if (!DOMsis.getElementById('monCss'))
            {
            var style = document.createElement("style");
            DOMsis.head.appendChild(style);
            style.id = 'monCss';
            //alert ('addCSS');
            style.sheet.insertRule("body  { background-color: #000000 !important;}");
            style.sheet.insertRule("label { color: #FFFFFF !important; }");
            style.sheet.insertRule(".item { width: 6rem !important;}");
            style.sheet.insertRule(".image-box { height: 6rem !important; min-width: 6rem; }");
            style.sheet.insertRule(".image     { opacity: 0.6;}");
            style.sheet.insertRule(".caption   { color: #FFFFFF !important; text-align: center;}");
            }
        }
    
    function init_radiocovid19() {
        //
        // Recherche l'IFRAME dans lequel l'élément 'streamlit-image-select' est chargé.
        // Puis ajoute une feuille de style pour surcharger quelques propriétés.
        //
        
        // Récupères les frames de la fenêtre
        let frames = window.parent.frames;
        // Parcourt les frames de la fenêtre à la recherche de la 'streamlit-image-select'
        for (i=0; i<frames.length; i++)
            {
            // Si le document contenu dans la frame a un titre
            if (frames[i].document.getElementsByTagName('title').length > 0) 
                {
                // et si ce titre est 'streamlit-image-select'
                if (frames[i].document.getElementsByTagName('title')[0].innerText === 'streamlit-image-select')
                    {
                    add_CSS(frames[i].document);
                    }
                }
            } 
        }
    
    // Attend la fin de chargement du document avant d'exécuter init_radiocovid19            
    if (window.parent.document.readyState === 'complete') 
        {
        init_radiocovid19()
        }
    
    </script>
    """
    html(script)  

###############################################################################

# Définition d'une fontion de pré-processing
def preprocessing (image,
                   mask=False,
                   masking=False,
                   cropping=False,
                   normalisation=False,
                   egalisation='CLAHE',
                   clahe_cliplimit=64):
    """Cette fonction réalise:
       - le Redimensionnement de l'image
       - Suppression des éventuels cadres noirs
       - Application du masque
       - Suppression des annotations
       - Egalisation des histogrammes
       - Application du Gaussian Blur
       - Normalisation

    Paramètres
    ----------
    image : str|np
        - si `str` : Url du image à pré-traiter
        - si `np`  : image à pré-traiter
    mask : str|tf.keras.models.Model
        - Si `str`   : Url du masque de l'image
        - Si `Model` : Modèle à utiliser pour prédire le masque
    masking = `False` :
        Si `False`,  le masque ne sera pas appliqué
        Si `True`, le masque sera appliqué
        Si `special`, un masque spécifique sera calculé qui conserve le centre
        des poumons.
    cropping = `False`
        Si `False`, l'image ne sera pas recadrée
        Si `True`, l'image sera recadrée autour de la région pulmonaire

    Retourne
    --------
    - image pré-processée

    """
    log = ''
    
    IMG_SHAPE=(256,256)

    # Chargement des images et des masks
    if isinstance(image, str):
        log += "Chargement de l'image\n"
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # S'assure que l'image est au format GRAYSCALE pour les images passées en numpy
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Si mask est un modèle
    if isinstance(mask, tf.keras.models.Model):

        log += 'Exécution modèle : Segmentation pulmonaire \n'
        # Calcule le masque en uitlisant le modèle
        i = image.astype('uint8')
        i = cv2.resize(i, (256,256))
        i = cv2.equalizeHist(i)
        i = cv2.GaussianBlur(i, ksize=(5,5), sigmaX=0)
        i = i.reshape(1,256,256,1)
        msk = mask(i)
        msk = tf.cast(msk > 0.5, 'float32')
        msk = msk.numpy().reshape(256,256)

    # Sinon, considère que mask est l'url de l'image du masque
    elif isinstance(mask, str):
        msk = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    else:
        msk = 255 * np.ones(IMG_SHAPE)

    # Si masking 'special', 
    # modifie le masque pour conserver la partie centrale des poumons
    if masking == 'special':
        # Pour chaque ligne de l'image
        for l in range(0, msk.shape[0]):
            # Récupère les index différents de zéro
            non_vides = msk[l,:].nonzero()[0]
            # Colorie les valeurs extrêmes avec du blanc pour garder le centre des poumons
            if len(non_vides) != 0:
                msk[l, non_vides.min():non_vides.max()] = msk.max()

    # Redimensionnement de l'image et du masque
    image = cv2.resize(image, dsize=IMG_SHAPE)
    msk   = cv2.resize(msk,   dsize=IMG_SHAPE)

    # Suppression des éventuels cadres noirs
    # Récupère les indices des pixels <> 0
    # X, Y = image.nonzero()
    # Fixe une limite de cadre
    # pour éviter de redimensionner toutes les images pour un bord de 1 px
    # if (X.min() < 5) and (X.max() > 250) and (Y.min() < 5) and (Y.max() > 250):
    #     pass
    # else:
    #     # Les valeurs min et max peuvent être utilisées pour déterminer la région qui comporte la radio
    #     roi_img = image[X.min():X.max(), Y.min():Y.max()]
    #     roi_msk = mask[X.min():X.max(), Y.min():Y.max()]
    #     image  = cv2.resize(roi_img, dsize=IMG_SHAPE)
    #     mask = cv2.resize(roi_msk, dsize=IMG_SHAPE)
    
    # Appliquer le masque
    if masking!=False:
        image = image * msk
        image = image.astype('uint8')

    # Recadrer sur la région pulmonaire
    roi = [0,0, 256,256]
    if cropping:
        log += 'Recadrage et zoom \n'
        Y, X = msk.nonzero()
        roi = {'x1':X.min(), 'y1':Y.min(), 'x2':X.max(), 'y2':Y.max()}
        roi_msk = msk[Y.min():Y.max(), X.min():X.max()]
        roi_img = image[Y.min():Y.max(), X.min():X.max()]
        msk   = cv2.resize(roi_msk, dsize=IMG_SHAPE)
        image  = cv2.resize(roi_img, dsize=IMG_SHAPE)
    
    # Suppression des annotations
    (_, img) = cv2.threshold(image, 248, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    inpaint_mask = cv2.drawContours(img, contours=contours, contourIdx=-1, color=255, thickness=12)
    # Remplace les zones annotées
    image = cv2.inpaint(image, inpaint_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)    
    
    # Egalisation
    if egalisation is True:
        image=cv2.equalizeHist(image)
    elif egalisation == 'CLAHE':
        log += 'Egalisation CLAHE'
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=clahe_cliplimit, tileGridSize=(8,8))
        image = clahe.apply(image)
    
    # Repasse l'image en 3 couches pour correspondre aux modèles préentrainés utilisés
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Etape finale de preprocessing selon le type de Core Model utilisé
    if normalisation == 'VGG16':
        image = tf.keras.applications.vgg16.preprocess_input(image)
    elif normalisation == 'VGG19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
    elif normalisation == 'Xception':
        image = tf.keras.applications.xception.preprocess_input(image)
    else:
        # Normalisation
        image = image / 255

    return image, log, roi


def gradCam(image : tf.Tensor, 
            model : tf.keras.Model,
            core_model_layername : str,
            last_conv_layername : str,
            pred_index : int|None = None,
            weighted_by_pred : bool = False,
            interpolation = cv2.INTER_CUBIC):
    """Génère une heatmap des zones considérées comme importantes par le modèle

    Paramètres
    ----------
    image : tf.Tensor
        Image à traiter sous la forme d'un tenseur (1, height, width, channels)
        Le type de données doit être 'float32'
    model : tf.keras.Model
        Modèle utilisé pour la classification
    core_model_layername : str = MODEL_CORE_MODEL
        Permet de spécifier le nom d'un sous modèle imbriqué.
        Utile dans le cas où `last_conv_layer` appratient à un sous modèle
        imbriqué (layer de type `functional`)
    last_conv_layername : str|None = None
        Nom de la dernière couche de convolution du modèle.
        Si `None`, la fonction va rechercher le dernier calque 
        de convolution du modèle.
    pred_index : int|None = None
        index de la classe à prédire.
        Si `None`, la heatmap retournée correspondra à la classe majoritaire
        calculée par le modèle
    weighted_by_pred : bool = False
        Si `True`, l'amplitude de la heatmap (0..255) sera pondérée 
        par la valeur de la prédiction (0..1).
        Par exemple, si la valeur d'un pixel est 200 et que la valeur 
        de la prédiction de la classe est 0.75, alors la valeur de ce pixel 
        sera 200 x 0.75 = 150 et non 200.
        Cela permet d'éviter d'avoir une heatmap avec des valeurs extrêmes (255) 
        si la prédiction de la classe n'est pas sûre (score proche de 0.5)
    interpolation : int = cv2.INTER_CUBIC
        Type d'interpolation à appliquer lorsque le gradCam est redimmensionné
        à la taille de l'image d'origine.

    Retourne
    --------
    Tuple contenant
    - Tableau numpy des points accordés à chaque pixel de l'image
    - Tableau des valeurs de prédiction
    """

    def get_layers(model) -> dict:
        """Retourne un dictionnaire {'nom layer':layer}
        des couches du modèle pasé en paramètre"""
        # liste toutes les couches du modèle
        layers = {}
        for l in model.layers:
            # Si la couche est un modèle
            if isinstance(l, tf.keras.Model):
                layers.update(get_layers(l))
            # Sinon, la couche est un calque
            else:
                layers[l.name] = l
        return layers

    # Récupère les couches du modèle ------------------------------------------
    layers = get_layers(model)
    last_conv_layer = layers[last_conv_layername]
    
    # Première étape, créer un nouveau modèle basé sur le modèle passé en paramètres
    # avec en retour : l'output du layer analysé, et l'output global du modèle (la prévision)
    # Si un 'core model' est utilisé (functionnal layer)
    # on fera d'abord appel au core model, puis le retour sera passé aux dernier
    # block du modèle général
    # modele : [VGG16][a][b] = [VGG16] puis [a][b] => [out]
    if core_model_layername is not None:
        core_model_layer = model.get_layer(core_model_layername)
        core_model = tf.keras.Model(core_model_layer.inputs, [last_conv_layer.output, core_model_layer.output])
        # Recherche le layer juste après le core_model_layername sélectionné
        n=[]
        for l in model.layers:
            n.append(l.name)
        # Reste du modèle sert à la classification
        clf_model = tf.keras.Model(model.get_layer(n[n.index(core_model_layername)+1]).input, [model.output])

    # Pas de Core Model
    else:
        # Modèle de classification
        clf_model = Model(model.inputs, [last_conv_layer.output, model.output])

    # Puis on calcule et récupère le gradient du calque surveillé et les prédictions du modèle
    with tf.GradientTape(persistent=True) as tape:

        # Si un core_model est déclaré
        if core_model_layername is not None:
            last_conv_layer_output, core_model_out = core_model(image, training=False)
            preds = clf_model(core_model_out, training=False)
        # Si pas de core model
        else:
            last_conv_layer_output, preds = clf_model(image, training=False)

        # S'il n'y a pas de classe particulière à surveiller,
        if pred_index is None:
            # On retourne les données de la classe déterminée par le modèle (classe majoritaire)
            pred_index = tf.argmax(preds[0])

        # Récupère la valeur de prédiction de la classe
        prediction = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    class_grads = tape.gradient(prediction, last_conv_layer_output)
       
    # Discard batch dim
    last_conv_layer_output = tf.squeeze(last_conv_layer_output)
    class_grads = tf.squeeze(class_grads)
    norm_grads = tf.divide(class_grads, tf.reduce_mean(tf.square(class_grads)) + tf.keras.backend.epsilon())

    # Compute weights
    weights = tf.reduce_mean(norm_grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, last_conv_layer_output), axis=-1)

    # Apply reLU
    cam = np.maximum(cam, 0)
    
    if (np.max(cam) != 0):
        cam = cam / np.max(cam)

    # Resize heatmap to image size
    cam  = cv2.resize(np.uint8(255*cam), (image.shape[1], image.shape[1]), interpolation=interpolation)

    # la heatmap est pondérée par le résultat de la prédiction
    if (weighted_by_pred):
        cam = cam * prediction.numpy()
        
    cam = np.uint8(cam)
    
    return cam, preds

def superimpose(image : np.ndarray, heatmap : np.ndarray, type='normal'):
    """Superpose un gradCam et une image pour facilier l'interprétation du modèle

    Paramètres
    ----------
    image : np.ndarray
        Image (width, height, 3) qui sera positionnée en arrière plan
    heatmap : np.ndarray
        heatmap (width, height) positionnée sur l'image.
    type : str = 'normal'
        Méthode de superposition d'image 
         - 'normal'    : la heatmap est simplement superposée à l'image
         - 'emphasize' : la heatmat est retraitée pour  accentuer les zones
           les plus chaudes.
         - 'retangle' : créé des rectangles autour des zones les plus 
            chaudes de la heatmap
    """
    # S'assure que la heatmap et l'image sont de même dimension
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Applique la transformation passée dans le paramètre `type`
    if type=='normal':
        cmap = cv2.COLORMAP_MAGMA
        
    elif type=='emphasize':
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap =  1 / (1 + np.exp(-50 * (heatmap-0.5)))
        cmap = cv2.COLORMAP_TURBO
        
    elif type=='rectangle':
        # Récupère les couleurs de la COLORMAP dans un tableau (256, 3)
        cmap = np.linspace(0,255,256, dtype='uint8')
        cmap = cv2.applyColorMap(cmap, cv2.COLORMAP_VIRIDIS)
        cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        cmap = cmap.reshape(256,3)
        color=tuple(map(int, cmap[heatmap.max()]))
        color=(235,245,255)
        
        thresh = heatmap.max() * 0.60
        _, threshold = cv2.threshold(heatmap, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
        contours, _  = cv2.findContours(threshold, mode=cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            image = cv2.rectangle(image, (x,y), ((x+w),(y+h)), color=color, thickness=1)
        return image

    else:
        raise ValueError(f"Le paramètre `type` doit avoir la valeur 'normal', 'emphasize' ou 'rectange'. Reçu : '{type}'")
        
    # Passe la heatmap sous la forme d'une image 0..255  
    if (heatmap.dtype.type is np.float_) and (0<=heatmap.max()<=1):
        heatmap = np.uint8(255 * heatmap)

    # Colorie la heatmap
    heatmap = cv2.applyColorMap(heatmap, cmap)
    
    # Finalement, superpose la heatmap avec l'image
    superimposed_img = heatmap * 0.7 + image
    superimposed_img = np.minimum(superimposed_img, 255).astype(np.uint8) 
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb