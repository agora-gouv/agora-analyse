import streamlit as st
import pandas as pd
import numpy as np
import os
import shlex
import matplotlib.pyplot as plt
import plotly.express as px
import collections


from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
print(path_root)
sys.path.append(str(path_root))

from assets.utils.dataload import load_cleaned_labels, load_doc_infos, read_csv_input, get_questions_df, read_sql_input
from assets.utils.datafilter import get_sentences_with_words, get_sentences_including_words
from assets.utils.dataprep import prep_sentiment_analysis, prep_doc_info
from assets.utils.dataclean import get_word_frequency
from assets.utils.dataviz import create_wordcloud_from_topic

TOPIC_FOLDER = "data/topic_modeling/"


# TODO: put in labeling pipeline 
# @st.cache_data
# def measure_similarity_of_topic(topic_labels: list[str], _topic_model):
#     embedding = _topic_model.embedding_model.embed(topic_labels)
#     similarity_matrix = cosine_similarity(embedding)
#     top_id = np.argmax(np.sum(similarity_matrix, axis=1))
#     top_label = topic_labels[top_id]

#     # scoring topic
#     triu_mat = np.triu(similarity_matrix, k=1)
#     score = np.mean(triu_mat[np.nonzero(triu_mat)])
#     return similarity_matrix, top_label, score


@st.cache_data
def get_doc_stats(doc_infos: pd.DataFrame, groupby="Topic")-> pd.DataFrame:
    doc_infos_upgraded = doc_infos.copy()
    doc_count = len(doc_infos_upgraded.index)
    threshold = 0.8
    doc_infos_upgraded["Good_proba"] = doc_infos_upgraded["Probability"] >= threshold
    stats = doc_infos_upgraded.groupby(groupby).agg(nb_doc=("Document", "count"), good_docs=("Good_proba", sum))
    stats["percentage"] = (stats["nb_doc"] / doc_count) * 100
    stats["percentage"] = stats["percentage"].round(decimals=2)
    stats["nb_doc"] = stats["nb_doc"].astype(int)
    return stats


@st.cache_resource
def plot_frequent_words(freq_words: pd.DataFrame, title="Fréquence des mots du topic"):
    color_sequence = px.colors.sequential.Viridis.copy()
    color_sequence.reverse()
    fig = px.bar(freq_words, x="freq", y="word", color="word", title=title, orientation='h', color_discrete_sequence=color_sequence)
    fig.update_layout(showlegend=False, title_x=0.3)
    fig.update_xaxes(title="Fréquence des mots")
    fig.update_yaxes(title="Mots importants")
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def plot_mattrix(data):
    fig, ax = plt.subplots(figsize = (2, 2))
    plt.imshow(data, interpolation='nearest')
    plt.xlabel("Label")
    plt.ylabel("Label")
    plt.title("Cosine Similarity")
    st.pyplot(fig)


def display_outliers_topic(doc_infos: pd.DataFrame, word_freq: pd.DataFrame, stats: pd.DataFrame, is_subtopic: bool=False):
    type = "Sous-topic" if is_subtopic else "Topic"
    st.write(f"#### Exploration du {type} des cas particuliers pour la question")
    title = f"{type} -1 : {int(stats.loc[-1]['nb_doc'])} réponses ({stats.loc[-1]['percentage']}%)"
    freq_words = word_freq[word_freq["topic"] == -1]
    plot_frequent_words(freq_words, title)
    display_answers_from_topic(doc_infos[doc_infos["Topic"] == -1])


def display_topic_overview(word_freq: pd.DataFrame, stats: pd.DataFrame, is_subtopic: bool=False):
    type = "Sous-topic" if is_subtopic else "Topic"
    st.write(f"#### Vue d'ensemble des {type}s générés pour la question")
    topic_range = min(len(stats.index)-1, 6)
    cols = st.columns(3)
    for topic in range(topic_range):
        with cols[topic%3]:
            title = f"{type} {topic} : {int(stats.loc[topic]['nb_doc'])} réponses ({stats.loc[topic]['percentage']}%)"
            freq_words_filter = word_freq[word_freq["topic"] == topic][["word", "freq"]]
            plot_frequent_words(freq_words_filter, title)
    return


def display_answers_from_topic(doc_info: pd.DataFrame):
    exp = st.expander("Afficher la liste les réponses du topic sélectionné")
    with exp:
        st.write("Légende : *(Score de confiance)(Score de sentiment) -- réponse*")
        st.write("'+': positif, '~': neutre, '-': négatif")
        st.dataframe(doc_info["Answer_with_proba"].values, use_container_width=True)
    return


@st.cache_data
def get_most_present_words_g(df: pd.DataFrame, col: str, ngram: int):
    c=collections.Counter()
    for x in df[col]:
        #x = i.rstrip().split(" ")
        c.update(set(zip(x[:-1],x[1:])))
    most_presents_bigram = list(map(lambda x: (" ".join(x[0]), x[1]), c.most_common(10)))
    most_presents_bigram = pd.DataFrame(most_presents_bigram, columns=["bigram", "count"])
    return most_presents_bigram


def subtopics_info(doc_infos: pd.DataFrame, topic: int):
    sub_doc_infos = doc_infos[doc_infos["Topic"] == topic]
    sub_stats = get_doc_stats(sub_doc_infos, "sub_topic")
    word_freq = get_word_frequency(sub_doc_infos, "Document", "sub_topic")
    display_topic_overview(word_freq, sub_stats, True)
    subtopic = st.selectbox("Sélectionnez le sous-topic à analyser : ", range(len(sub_stats) -1))
    subtopic_info = sub_doc_infos[sub_doc_infos["sub_topic"] == subtopic]
    most_presents_bigram = get_most_present_words_g(subtopic_info, "tokens", 2)
    st.write("##### Bi-gram les plus présents dans le sous-topic :")
    st.write(most_presents_bigram)
    display_selected_bigrams(subtopic_info, most_presents_bigram)
    display_answers_from_topic(subtopic_info)
    

def display_topic_basic_info(topic: int, cleaned_labels: pd.DataFrame, word_freq: pd.DataFrame, stats: pd.DataFrame):
    #sim_matrix, top_label, score = measure_similarity_of_topic(cleaned_labels[topic], custom_bertopic)
    st.write("#### Infos sur le topic")
    nb_doc = stats.loc[topic]["nb_doc"]
    percentage = stats.loc[topic]["percentage"]
    st.write(f"Nombre de documents : **{int(nb_doc)}** *({str(percentage)}%)*")
    st.write("Meilleur label généré d'après le modèle : ")
    st.write(cleaned_labels.loc[topic]["label"])
    #st.write("**" + top_label + "**")
    #st.write("*Score de confiance : " + str(score) + "*")


    # TODO : Check labels from SQL Data
    # other_labels = st.checkbox("Afficher les labels potentiels ?")
    # if other_labels:
    #     st.write("Les labels potentiels sont : ")
    #     for i in range(len(cleaned_labels[topic])):
    #         st.write("*" + cleaned_labels[topic][i] + "*")
    
    plot_frequent_words(word_freq[word_freq["topic"] == topic])


def display_sentiment_analysis(df_sentiment: pd.DataFrame):
    st.write("## Analyse de sentiments :")
    st.write("Score de sentiment par topic : ")
    # TODO: put in sentiment pipeline V
    topics_sentiments = df_sentiment.groupby(["Topic", "sentiment"]).agg(score_sum=("sentiment_score", sum), high_score_count=("is_high_score", sum), count=("sentiment", "count")).reset_index()
    color_map = {"positive": "green", "neutral": "blue", "negative": "red"}
    st.dataframe(df_sentiment[["Document", "sentiment", "sentiment_score"]], use_container_width=True, hide_index=True)
    fig = px.bar(topics_sentiments, "Topic", "score_sum", color="sentiment", title="Analyse de sentiments par topic", color_discrete_map=color_map)
    fig.update_layout(showlegend=False, title_x=0.3)
    fig.update_xaxes(title="Topics")
    fig.update_yaxes(title="Score de sentiment")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(topics_sentiments, "Topic", "high_score_count", color="sentiment", title="Analyse de sentiments par topic", color_discrete_map=color_map)
    fig.update_layout(showlegend=False, title_x=0.3)
    fig.update_xaxes(title="Topics")
    fig.update_yaxes(title="Nombre de contribution forte")
    st.plotly_chart(fig, use_container_width=True)


def display_selected_bigrams(topic_info: pd.DataFrame, most_presents_bigram: pd.DataFrame):
    selected_bigram = st.selectbox("Selectionner le bigram dont vous voulez voir les contributions", most_presents_bigram["bigram"].values)
    sentences = get_sentences_with_words(topic_info, "tokens", selected_bigram, "Document")
    
    with st.expander("Réponses avec le bigram sélectionné"):
        st.dataframe(sentences.values, use_container_width=True)


def display_topic_info(topic: int, doc_infos: pd.DataFrame, cleaned_labels: list[list[str]], word_freq: pd.DataFrame, question_short: str):
    stats = get_doc_stats(doc_infos)
    if topic is not None:
        topic_info = doc_infos[doc_infos["Topic"] == topic].copy()
        st.write("### Topic " + str(topic))
        label_cols = st.columns(2)
        with label_cols[0]:
            display_topic_basic_info(topic, cleaned_labels, word_freq, stats)
            
        with label_cols[1]:
            wc_folder = TOPIC_FOLDER + question_short + "/wordcloud/"
            wc_filepath = wc_folder + "wc_" + str(topic) + ".png"
            st.image(wc_filepath, use_column_width=True)
            most_presents_bigram = get_most_present_words_g(topic_info, "tokens", 2)
            st.write("##### Bi-gram les plus présents dans le topic :")
            st.write(most_presents_bigram)
        
        display_selected_bigrams(topic_info, most_presents_bigram)
        best_answers = doc_infos[(doc_infos["Topic"] == topic) & (doc_infos["Representative_document"])]["Document"].values
        #best_answers = shlex.split(best[1:-1])
        expander = st.expander("Afficher les réponses pertinentes")
        for i in range(len(best_answers)):
            expander.write(best_answers[i])
        display_answers_from_topic(topic_info)
        
        subtopics_info(doc_infos, topic)


def topic_selection(doc_infos: pd.DataFrame, word_freq: pd.DataFrame, cleaned_labels: pd.DataFrame, question_short: str):
    topic_count = 8
    label_tab, wc_tab, outlier_tab, sentiment_tab = st.tabs(["Détails des Topics", "Nuages de mots", "Cas Particuliers", "Analyse de sentiment"])
    st.markdown("---")
    with wc_tab:
        force_compute = st.button("Recalculer les nuages de mots")
        wc_folder = TOPIC_FOLDER + question_short + "/wordcloud/"
        wc_columns = st.columns(4)
        for i in range(topic_count):
            os.makedirs(wc_folder, exist_ok=True)
            wc_filepath = wc_folder + "wc_" + str(i) + ".png"
            # Si les nuages de mots n'existent pas les calculer
            if not os.path.isfile(wc_filepath) or force_compute:
                wordcloud = create_wordcloud_from_topic(word_freq[word_freq["topic"] == i])
                wordcloud.to_file(wc_filepath)
            # Afficher les nuages de mots
            with wc_columns[i%4]:
                st.write("### Topic " + str(i))
                st.image(wc_filepath, width=300)
    with label_tab:
        topic = st.selectbox("Sélectionnez le topic à analyser : ", range(len(cleaned_labels)))
        display_topic_info(topic, doc_infos, cleaned_labels, word_freq, question_short)
    with outlier_tab:
        stats = get_doc_stats(doc_infos)
        display_outliers_topic(doc_infos, word_freq, stats)
    with sentiment_tab:
        df_sentiment = prep_sentiment_analysis(doc_infos)
        if df_sentiment is not None:
            display_sentiment_analysis(df_sentiment)
        else:
            st.write("Pas d'analyse de sentiment disponible pour cette question pour le moment.")


def select_question_from_database(questions_df):
    options = questions_df["id"].values
    selected_option = st.selectbox("Choisissez la question dont vous voulez voir l'analyse :", options=options)
    #question_id = questions_df[questions_df["title"] == selected_option]["id"][0]
    st.write(selected_option)
    return selected_option


def write():
    st.write("## Evaluation des topics générés")
    
    analyse = st.radio("Choisissez le mode d'analyse :", options=["Par fichier", "Par SQL"])
    if analyse == "Par fichier": 
        doc_infos_raw = read_csv_input()
        
    else:
        questions_df = get_questions_df()
        question_id = select_question_from_database(questions_df)
        doc_infos_raw = read_sql_input(question_id)

    question_short = "Custom_analysis"
    
    
    if doc_infos_raw is not None:
        doc_infos = prep_doc_info(doc_infos_raw)
        #cleaned_labels = load_cleaned_labels(question_short, TOPIC_FOLDER)
        cleaned_labels = doc_infos.groupby("Topic").agg(label=("Name", "first"))
        stats = get_doc_stats(doc_infos)
        
        word_freq = get_word_frequency(doc_infos, "Document", "Topic")
        display_topic_overview(word_freq, stats)
        
        
        topic_selection(doc_infos, word_freq, cleaned_labels, question_short)
    return


if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📊", page_title="Agora -- NLP"
    )
    write()
