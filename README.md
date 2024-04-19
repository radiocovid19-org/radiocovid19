# Analyse de radiographies pulmonaires Covid-19


## Présentation et Installation


Notre projet s'inscrit dans le cadre de la formation [Data Science](https://datascientest.com/en/data-scientist-course) dispensée par [DataScientest](https://datascientest.com/).

L'objectif étant de  **developper un modèle de deep learning permettant de détecter plus facilement les cas positifs covid en classifiant les images radiographiques pulmonaires.**

Le jeu de données est composé de 21.165 radiographies thoraciques et leurs masques. Il est mis à disposition [ici](./data).
 

Les membres d'équipe:

- Frédéric Ferchaud ([GitHub](https://github.com/Fred-FR44) / [LinkedIn](https://www.linkedin.com/in/frederic-ferchaud/))
- Soumaya Jendoubi Elhabibi ([GitHub](https://github.com/Soumaya-JE) / [LinkedIn](https://www.linkedin.com/in/soumaya-jendoubi-273ba4a4/))

Pour plus de détails sur le code,veuillez consulter les [notebooks](./notebooks). 

Les conditions requises pour l'installation peuvent être mises en place à travers :
```
pip install -r requirements.txt
```

## Streamlit App

Afin de mieux utiliser l'application,il faut télécharger les images et les modèles disponibles sur [assets](./streamlit_app/assets)

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

L'application sera disponible à [localhost:8501](http://localhost:8501).
