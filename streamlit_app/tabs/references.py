import streamlit as st
import cv2


title = "Références"
sidebar_name = "Références"


def run():

    st.title(title)

    st.markdown("## Dataset")
    st.markdown(""" 
- Kaggle  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

- M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam,
M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam,  
“Can AI help in screening Viral and COVID-19 pneumonia?” - IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.  
https://ieeexplore.ieee.org/document/9144185

- Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury,  
“Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images.”  
https://doi.org/10.1016/j.compbiomed.2021.104319
""")
    
    st.markdown("## Images")
    
    st.markdown("#### Section : Introduction")
    st.markdown("""
- Bannière  
<a href="https://fr.freepik.com/photos-gratuite/chercheur-asiatique-verifie-poitrine-pulmonaire-du-rapport-radiographie-du-patient-covid19-resultat-positif-infection-detech-analyse-epidemique-epidemie-virus-corona-effet-du-covid-dans-concept-test-du-corps-humain_25118289.htm#fromView=search&page=1&position=2&uuid=a9dfed14-4c94-4cc9-8eee-2b1b751d8752">Image de Lifestylememory sur Freepik</a>
""", unsafe_allow_html=True)
    st.markdown("""
- Poumons avec GGO  
Miao, Liyun & Cai, Hou-Rong. (2009).  
“Cystic changes in mucosa-associated lymphoid tissue lymphoma of lung: A case report.”  
Chinese medical journal. 122. 748-51. 10.3760/cma.j.issn.0366-6999.2009.06.030. 

- Localisation, Opacités pulmonaires
Ruchi Yadav, MD, Debasis Sahoo, MD, FCCP and Ruffin Graham, MD  
“Thoracic imaging in COVID–19, Cleveland Clinic Journal of Medicine June 2020”  
DOI: https://doi.org/10.3949/ccjm.87a.ccc032

""")

    st.markdown("#### Section : RadioCovid19")
    st.markdown("""
- Images du dataset  
Kaggle
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Images extérieures au dataset  
Radiology materclass
https://www.radiologymasterclass.co.uk/
""")
    
    
    st.write("   ")
    st.markdown("---")
    st.markdown("<div style='padding-bottom: 10rem;'></div>", unsafe_allow_html=True)
    
    
    
    

    
