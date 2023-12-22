import pandas as pd


def get_sentences_with_words(df: pd.DataFrame, col: str, words: str, returned_col: str):
    mask = df[col].apply(lambda x: words in " ".join(x))
    sentences = df[mask][returned_col]
    return sentences

# Function to get responses that contains a set of words
def get_sentences_including_words(df: pd.DataFrame, col: str, words: str, col_returned: str):
    mask = df[col].apply(lambda x: all(word in x for word in words))
    df1 = df[mask]
    return df1[col_returned]