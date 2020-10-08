# import numpy as np
# from scipy import spatial
# from sklearn.manifold import TSNE
#
# embeddings_dict = {}
# with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector
#
# def find_closest_embeddings(embedding):
#     return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
#
# print(find_closest_embeddings(embeddings_dict["king"])[1:6])
#
# print(find_closest_embeddings(
#     embeddings_dict["twig"] - embeddings_dict["branch"] + embeddings_dict["hand"]
# )[:5])
#
# tsne = TSNE(n_components=2, random_state=0)
# words = list(embeddings_dict.keys())
# vectors = [embeddings_dict[word] for word in words]
# Y = tsne.fit_transform(vectors[:1000])
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle

data = pd.read_excel(r"G:\NLP\hw1\opinion_mining\P1_training.xlsx", encoding='unicode_escape', header=0, names=["text", "opinion"],
                     error_bad_lines=False, lineterminator='\n')
test_data = pd.read_excel(r"G:\NLP\hw1\opinion_mining\P1_testing.xlsx", encoding='unicode_escape', header=0, names=["test_text", "test_opinion"],
                     error_bad_lines=False, lineterminator='\n')
text = data['text']
label = data['opinion']

test_text = test_data['test_text']
test_label = test_data['test_opinion']


train_cutoff = int(len(text))
corpus = text
corpus = corpus.append(test_text)
nlp = spacy.load('en_core_web_md')

spacy_prediction_2=[]
spacy_prediction_1=[]
spacy_prediction_3=[]

def bigram():
    spacy_features = []
    for line in corpus:
        sentence_vector = 0
        words = line.split()
        for i in range(2, len(words)):
            word_vec = nlp(words[i]+" "+words[i-1])
            sentence_vector += word_vec.vector
        print(sentence_vector)
        sentence_vector /= (len(words)-2)
        spacy_features.append(sentence_vector)

    svc = LinearSVC()
    spacy_word_embed = CalibratedClassifierCV(svc)
    spacy_word_embed.fit(spacy_features[0: train_cutoff], label)

    pickle.dump(spacy_word_embed, open("G:/NLP/hw1/opinion_mining/bigram_proposed", 'wb'))
    # spacy_word_embed = pickle.load(open("G:/NLP/hw1/opinion_mining/bigram_proposed", 'rb'))

    spacy_prediction_2 = spacy_word_embed.predict_proba(spacy_features[train_cutoff: len(corpus)])  # pass matrix
    print("spacy_prediction for 2-gram : ", spacy_prediction_2)
    f1_spacy=f1_score(spacy_prediction_2, test_label, average='macro')
    print(f1_spacy)
    acc_spacy=accuracy_score(spacy_prediction_2, test_label)
    print(acc_spacy)

# ___
def trigram():
    spacy_features = []
    for line in corpus:
        sentence_vector = []
        words = line.split()
        for i in range(3, len(words)):
            word_vec = nlp(words[i] + " " + words[i - 1]+" "+words[i - 2])
            sentence_vector += word_vec.vector
        print(sentence_vector)
        sentence_vector /= (len(words)-3)
        spacy_features.append(sentence_vector)

    svc = LinearSVC()
    spacy_word_embed = CalibratedClassifierCV(svc)
    spacy_word_embed.fit(spacy_features[0: train_cutoff], label)

    pickle.dump(spacy_word_embed, open("G:/NLP/hw1/opinion_mining/trigram_proposed", 'wb'))
    # spacy_word_embed = pickle.load(open("G:/NLP/hw1/opinion_mining/trigram_proposed", 'rb'))

    spacy_prediction_3 = spacy_word_embed.predict_proba(spacy_features[train_cutoff: len(corpus)])  # pass matrix
    print("spacy_prediction for 3-gram : ", spacy_prediction_3)
    f1_spacy = f1_score(spacy_prediction_3, test_label, average='macro')
    print(f1_spacy)
    acc_spacy = accuracy_score(spacy_prediction_3, test_label)
    print(acc_spacy)


# ___
def unigram():
    spacy_features = []
    for line in corpus:
        sentence_vector = 0
        words = line.split()
        for i in range(0, len(words)):
            word_vec = nlp(words[i])
            sentence_vector += word_vec.vector
        print(sentence_vector)
        sentence_vector /= len(words)
        spacy_features.append(sentence_vector)
    svc = LinearSVC()
    spacy_word_embed = CalibratedClassifierCV(svc)
    spacy_word_embed.fit(spacy_features[0: train_cutoff], label)

    pickle.dump(spacy_word_embed, open("G:/NLP/hw1/opinion_mining/unigram_proposed", 'wb'))
    # spacy_word_embed = pickle.load(open("G:/NLP/hw1/opinion_mining/trigram_proposed", 'rb'))

    spacy_prediction_1 = spacy_word_embed.predict_proba(spacy_features[train_cutoff: len(corpus)])  # pass matrix
    print("spacy_prediction for 3-gram : ", spacy_prediction_3)
    f1_spacy = f1_score(spacy_prediction_1, test_label, average='macro')
    print(f1_spacy)
    acc_spacy = accuracy_score(spacy_prediction_1, test_label)
    print(acc_spacy)


import statistics
unigram()
bigram()
trigram()
final_rank_prediction = []
final_confidence_prediction = []
for i in range(0, len(spacy_prediction_1)):
    pred1 = np.where(spacy_prediction_1[i] == np.max(spacy_prediction_1[i]))
    pred2 = np.where(spacy_prediction_2[i] == np.max(spacy_prediction_2[i]))
    pred3 = np.where(spacy_prediction_3[i] == np.max(spacy_prediction_3[i]))
    print([pred1[0][0], pred2[0][0], pred3[0][0]])
    pred_i = statistics.mode([pred1[0][0], pred2[0][0], pred3[0][0]])
    final_rank_prediction.append(pred_i)
    all_spacy = spacy_prediction_1[i] + spacy_prediction_2[i] + spacy_prediction_3[i]
    max_index = np.where(all_spacy == np.max(all_spacy))
    max_index = max_index[0][0] % 3
    final_confidence_prediction.append(max_index)

f1_proposed = f1_score(final_rank_prediction, label[train_cutoff + 1: len(text)], average='macro')
print("f1_final_rank_prediction", f1_proposed)

acc_spacy = accuracy_score(final_rank_prediction, label[train_cutoff + 1: len(text)])
print("accuracy_final_rank_prediction", acc_spacy)

f1_proposed = f1_score(final_confidence_prediction, label[train_cutoff + 1: len(text)], average='macro')
print("final_confidence_prediction", f1_proposed)

acc_spacy = accuracy_score(final_confidence_prediction, label[train_cutoff + 1: len(text)])
print("final_confidence_prediction", acc_spacy)
