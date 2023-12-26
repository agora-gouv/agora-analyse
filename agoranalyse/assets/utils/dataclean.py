import pandas as pd
import string
import collections
from nltk.corpus import stopwords
import streamlit as st




STOPWORDS = stopwords.words("french")
PUNCTUATION = string.punctuation


def remove_punctuation(input_string):
    for char in PUNCTUATION:
        input_string = input_string.replace(char, ' ')
    return input_string


def remove_stopwords(input_string):
    return ' '.join([word for word in input_string.lower().split() if word not in STOPWORDS])


def preprocess(input_string):
    no_punctuation = remove_punctuation(input_string)
    no_stopwords = remove_stopwords(no_punctuation)
    return no_stopwords


### ------------- Token Count -----------------
@st.cache_data
def get_word_frequency(df: pd.DataFrame, text_col: str, groupby_col: str):
    df['clean'] = df[text_col].apply(preprocess)
    group_counters = dict()
    freq = []
    for groupby_key, group in df.groupby(groupby_col):
        group_counters[groupby_key] = pd.Series(' '.join(group["clean"]).split()).value_counts()[:10]
        size = len(group["clean"].index)
        for i in range(10):
            freq.append(group_counters[groupby_key][i] / size)

    counter_df = pd.concat(group_counters)
    counter_df = counter_df.reset_index().rename(columns={"level_0": "topic", "level_1": "word", 0: "count"})
    counter_df["freq"] = freq
    return counter_df
