from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

file_name = "G:\\NLP\\hw1\\training_pasted.xlsx"
data = pd.read_excel(r"G:\NLP\hw1\opinion_mining\P1_training.xlsx", encoding='unicode_escape', header=0, names=["text", "opinion"],
                     error_bad_lines=False, lineterminator='\n')
test_data = pd.read_excel(r"G:\NLP\hw1\opinion_mining\P1_testing.xlsx", encoding='unicode_escape', header=0, names=["test_text", "test_opinion"],
                     error_bad_lines=False, lineterminator='\n');
text = data['text']
label = data['opinion']

test_text = test_data['test_text']
test_label = test_data['test_opinion']


train_cutoff = int(len(text))
corpus = text
corpus = corpus.append(test_text)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def tf_idf_baseline():
    vectoriser = TfidfVectorizer()
    doc_term_matrix = vectoriser.fit_transform(corpus)

    # Converting the labels from strings to binary
    # tfidf_model = LinearSVC()
    # tfidf_model.fit(doc_term_matrix[0: train_cutoff], label)
    # pickle.dump(tfidf_model, open("G:/NLP/hw1/opinion_mining/tfidf_baseline", 'wb'))

    tfidf_model = pickle.load(open("G:/NLP/hw1/opinion_mining/tfidf_baseline", 'rb'))

    tfidf_prediction = tfidf_model.predict(doc_term_matrix[train_cutoff: len(corpus)])  # pass matrix
    f1_tfidf = f1_score(tfidf_prediction, test_label, average='macro')
    print("tfidf F1 score : ",f1_tfidf)
    acc_tfidf = accuracy_score(tfidf_prediction, test_label)
    print("tfidf accuracy score : ",acc_tfidf)


# print(tfidf_prediction)
import spacy


# Load the spacy model that you have installed
def spacy_baseline():
    nlp = spacy.load('en_core_web_md')
    spacy_features = []
    for line in corpus:
        doc = nlp(line)
        spacy_features.append(doc.vector)
    # spacy_word_embed = LinearSVC()
    # spacy_word_embed.fit(spacy_features[0: train_cutoff], label)
    #pickle.dump(spacy_word_embed, open("G:/NLP/hw1/opinion_mining/spacy_baseline", 'wb'))

    spacy_word_embed = pickle.load(open("G:/NLP/hw1/opinion_mining/spacy_baseline", 'rb'))

    spacy_prediction = spacy_word_embed.predict(spacy_features[train_cutoff: len(corpus)])  # pass matrix
    f1_spacy = f1_score(spacy_prediction, test_label, average='macro')
    print("spacy F1 score : ",f1_spacy)
    acc_spacy = accuracy_score(spacy_prediction, test_label)
    print("spacy accuracy score : ",acc_spacy)


tf_idf_baseline()
spacy_baseline()
