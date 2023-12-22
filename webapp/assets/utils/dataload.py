import pandas as pd
import streamlit as st
from bertopic import BERTopic
import json
import os


def read_csv_input():
    uploaded_file = st.file_uploader("Enter doc_infos file here")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None



@st.cache_data
def load_model(model_path: str)-> BERTopic:
    custom_bertopic = BERTopic.load(model_path)
    return custom_bertopic


@st.cache_data
def load_cleaned_labels(question_name: str, folder: str)-> list[list[str]]:
    with open(folder + question_name + "/cleaned_labels.json") as f:
        data = json.load(f)
    return data


@st.cache_data
def load_doc_infos(filepath: str)-> pd.DataFrame:
    if os.path.exists(filepath):
        doc_infos = pd.read_csv(filepath, index_col=0)
        doc_infos["Topic"] = doc_infos["Topic"].astype(int)
        doc_infos = doc_infos.sort_values("Probability", ascending=False)
        return doc_infos
    return None


@st.cache_data
def load_stat_dict(question_name: str, folder: str):
    filepath = folder + question_name + "/stat_dict.json"
    data = {}
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
    return data
