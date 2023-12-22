import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def tokenize(vect, bow, col_name="count", token_col="word") -> pd.DataFrame:
    df_tokens = pd.DataFrame(vect.get_feature_names_out(), columns=[token_col])
    df_tokens[col_name] = bow.sum(axis=0).tolist()[0]
    return df_tokens


def generate_vectorizer(ngram_range=(1,1), tokenizer = lambda x: x):
    max_features = 100000
    min_df = 1
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df,
                                            tokenizer=tokenizer,
                                            stop_words=None, lowercase=False)
    return vectorizer


def get_most_present_words(tokens: pd.DataFrame, count:int, col_name="count", token_col="word"):
    return tokens.sort_values(col_name, ascending=0).head(count)[[token_col, col_name]]


def most_common_2g(df_data: pd.DataFrame, text_col: str, most_present_word_count = 5):
    vect_2g = generate_vectorizer((2,2))
    bow_2g = vect_2g.fit_transform(df_data[text_col])
    df_tokens_2g = tokenize(vect_2g, bow_2g)
    
    most_common_words_2g = get_most_present_words(df_tokens_2g, most_present_word_count)
    return most_common_words_2g