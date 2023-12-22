from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
import streamlit as st
from PIL import Image
import awesome_streamlit as ast
from assets.pages import Evaluation_des_topics, About

#from dotenv import load_dotenv
#load_dotenv()


PAGES = {
    "❓ A propos": About,
    "📊 Evaluations des topics": Evaluation_des_topics,
    #"📈 Calcul du Qualiscore": Qualiscore
}


def main():
    """Main function of the app"""

    # logo = Image.open("src/webapp/assets/medias/logo.png")
    # st.sidebar.image(logo, width=300)
    st.sidebar.title("Navigation Agora NLP")
    selection = st.sidebar.radio("Choisir la page", list(PAGES.keys()))
    page = PAGES[selection]
    st.sidebar.markdown("""---""")
    with st.spinner(f"Chargement {selection} ..."):
        ast.shared.components.write_page(page)

    columns = st.sidebar.columns(2)
    if columns[0].button("Quitter l'Application"):
        os._exit(1)

    st.sidebar.info(
        """
        Info : Ceci est un POC
        """
    )


if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📚", page_title="Agora -- NLP"
    )
    
    main()
