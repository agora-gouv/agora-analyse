import streamlit as st

IMAGE_FOLDER = "agora_topic_modeling/webapp/assets/images/"

def computing_infos():
    st.write("#### Information sur les calculs")
    st.write("Pour évaluer la qualité des labels que nous attribuons aux différents topics identifiés par nos modèles, nous allons comparer les différents labels entre eux :")
    st.write("- comparaison de la similarité : le label le plus proche des autres labels généré est moins spécifique et donc le plus général")
    st.write("- score de confiance")


def write():
    st.write("## A propos de cette WebApp")
    st.write("Cette webapp s'inscrit dans le projet Agora NLP : **Synthèse des réponses aux questions ouvertes à l'aide du NLP** *(Natural Language Processing)*")
    col1, col2 = st.columns(2)
    col1.image(IMAGE_FOLDER + "Archi_sans_webapp.png")
    
    st.write("Elle sert d'intermédiaire entre la création de synthèse par les pipelines *Airflow* et les dashboard sur Metabase")

    computing_infos()
    #st.write("Il est important de garder une étape d'intervention humaine dans ce genre de projet car ")
    return


if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📊", page_title="Agora -- NLP"
    )
    write()