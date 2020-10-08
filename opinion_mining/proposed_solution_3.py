import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pickle

def preproces(corpus):
    ans=[]
    for text in corpus:
        sentence=""
        words = text.split()
        words = [word.lower() for word in words]
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in stripped if not w in stop_words]
        sentence=" ".join(words)
        ans.add(sentence)
    return ans


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
tf.disable_eager_execution()
tf.disable_v2_behavior()

embed = hub.Module(module_url)

data = pd.read_excel(r"G:/NLP/hw1/opinion_mining/P1_training.xlsx", encoding='unicode_escape', header=0, names=["text", "opinion"],error_bad_lines=False, lineterminator='\n')
test_data = pd.read_excel(r"G:/NLP/hw1/opinion_mining/P1_testing.xlsx", encoding='unicode_escape', header=0, names=["test_text", "test_opinion"],
                     error_bad_lines=False, lineterminator='\n')
text = data['text']
label = data['opinion']

test_text = test_data['test_text']
test_label = test_data['test_opinion']

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(text))
  test_embeddings = session.run(embed(test_text))

  #model = LinearSVC()
  #model.fit(message_embeddings, label)

  #pickle.dump(model, open("G:/NLP/hw1/opinion_mining/universal", 'wb'))
  model = pickle.load(open("G:/NLP/hw1/opinion_mining/universal", 'rb'))

  prediction = model.predict(test_embeddings)  # pass matrix
  f1_proposed = f1_score(prediction, test_label, average='macro')
  print("universal F1 score : ",f1_proposed)
  acc_proposed = accuracy_score(prediction, test_label)
  print("universal accuracy score : ",acc_proposed)
