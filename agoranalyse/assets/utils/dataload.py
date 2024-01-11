import pandas as pd
import numpy as np
import streamlit as st
import json
import os
from sqlalchemy import create_engine
from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
print(path_root)
sys.path.append(str(path_root))

from assets.utils.psql_utils import get_connection


def get_questions_df():
    #conn = get_connection(section="agora_nlp_psy")
    url = os.getenv("SCALINGO_POSTGRESQL_URL")
    engine = create_engine(url)
    with engine.connect() as conn:
        questions_df = pd.read_sql_query("SELECT * FROM questions", con=conn)
    #conn.close()
    return questions_df


def read_representative_docs(topics_ids: list[str], conn)-> pd.DataFrame:
    separator = "', '"
    query = f"SELECT * FROM representative_responses WHERE topic_id IN ( '{separator.join(topics_ids)}' )"
    url = os.getenv("SCALINGO_POSTGRESQL_URL")
    engine = create_engine(url)
    with engine.connect() as conn:
        representative_df = pd.read_sql_query(query, con=conn)
    representative_df = representative_df.drop(axis=0, columns="topic_id")
    return representative_df


def read_sql_input(question_id: str):
    
    
    url = os.getenv("SCALINGO_POSTGRESQL_URL")
    engine = create_engine(url)
    with engine.connect() as conn:
        topic_query = f"SELECT * FROM topics WHERE question_id='{question_id}'"
        topics_df = pd.read_sql_query(topic_query, con=conn)
        topics_ids = topics_df["id"]
        sub_topics = topics_df[~topics_df["parent_topic_id"].isna()]
        separator = "', '"
        query = f"SELECT * FROM responses WHERE topic_id IN ( '{separator.join(topics_ids)}' )"
        doc_info_raw = pd.read_sql_query(query, con=conn)
    representative_df = read_representative_docs(topics_ids, conn)
    representative_df["Representative_document"] = True
    doc_with_topics = doc_info_raw.merge(topics_df, left_on="topic_id", right_on="id", suffixes=("", "topic"))
    doc_with_topics = doc_with_topics.merge(sub_topics, left_on="sub_topic_id", right_on="id", suffixes=("", "sub"))
    doc_with_topics = representative_df.merge(doc_with_topics, left_on="response_id", right_on="id", how="right", suffixes=("", "response"))
    doc_with_topics["Topic"] = doc_with_topics["name"].str.split("_").str[0].astype(int)
    doc_with_topics["namesub"] = doc_with_topics["namesub"].fillna("-2_not_subtopic")
    doc_with_topics["sub_topic"] = np.where(~doc_with_topics["namesub"].isna(), doc_with_topics["namesub"].str.split("_").str[0].astype(int), -2)
    rename = {"text": "Document",
              "topic_probability": "Probability",
              "name": "Name"}
    doc_with_topics["Representative_document"] = doc_with_topics["Representative_document"].fillna(False)
    doc_with_topics = doc_with_topics.drop(axis=0, columns=["idtopic"])
    doc_with_topics = doc_with_topics.rename(columns=rename)
    return doc_with_topics



def read_csv_input():
    uploaded_file = st.file_uploader("Enter doc_infos file here")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None


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
