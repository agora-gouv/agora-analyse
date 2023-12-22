import pandas as pd
import streamlit as st
from nltk.corpus import stopwords


@st.cache_data
def prep_sentiment_analysis(df_sentiment: pd.DataFrame)-> pd.DataFrame:
    if df_sentiment is None:
        return None
    df_sentiment["is_high_score"] = df_sentiment["score"] > 0.75
    return df_sentiment


@st.cache_data
def remove_stopwords(sentence: str, stopwords: list[str])-> list[str]:
    tokens = sentence.lower().split(" ")
    result = []
    for token in tokens:
        if token not in stopwords:
            result.append(token)
    return result


@st.cache_data
def get_tokens_without_stopwords(df: pd.DataFrame, col: str)-> pd.DataFrame:
    stop_words=stopwords.words("french")
    df["tokens"] = df["Document"].apply(lambda x: remove_stopwords(x, stop_words))
    return df


@st.cache_data
def prep_doc_info(doc_infos: pd.DataFrame):
    doc_infos_prepped = doc_infos.copy()
    doc_infos_prepped["Answer_with_proba"] = "(" + doc_infos_prepped["Probability"].astype(str) + ") " + doc_infos_prepped["Document"] 
    doc_infos_prepped = get_tokens_without_stopwords(doc_infos_prepped, "Document")
    return doc_infos_prepped
