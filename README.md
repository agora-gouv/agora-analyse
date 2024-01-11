# Agoranalyse
Application Web de visualisation pour les analyses de verbatim.
Projet réalisé dans le cadre de l'automatisation de l'analyse de réponses à question ouverts de la startup d'état **AGORA**

## Structure du répertoire

- **agoranalyse** : dossier contenant le code pour la webapp
  - **assets/pages** : dossier du code des pages
  - **assets/utils** : dossier de code utilisé par les page notamment pour récupérer des données ou faire du traitement de données avant leur affichage
- **Procfile** : fichier utile au déploiement sur Scalingo
- **requirement.txt** : fichier contenant les dépendances des librairies pythons

## Pré-requis

- Avoir installé python

## Installation

L'installation est un processus en 3 étapes :
- Crée un nouvel **environement virtuel** *python* (*aka venv*)
  - Cette étape est seuelement nécessaire la première fois que vous créez un *venv*
- Activer **l'environnement virtuel** *python*
- Installer les dépendances requises

```bash 
# Create a new environment named as you would like 
python3 -m venv new_env
```

```bash 
# Load the virtual environment you created previously
source new_env/bin/activate
```

```bash
# Install the required dependencies that have been stored in a file for the occasion
pip install -r requirements.txt
```

### Usage
Pour lancer l'application web, vous pouvez exécuter la commande suivante :
```bash
streamlit run agoranale/webapp.py
```


> Author: Theo Santos