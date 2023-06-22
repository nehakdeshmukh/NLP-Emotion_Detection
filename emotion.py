
import pandas as pd
import nltk
import pandas as pd
import neattext.functions as nfx
import numpy as np
import seaborn as sns   # Load Data Viz Pkgs
import matplotlib.pyplot as plt
import os
import re
import csv
import spacy
import pickle
import string

from spacy.lang.en.examples import sentences
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split      #Split data
from sklearn.metrics import precision_score,recall_score,accuracy_score,classification_report,confusion_matrix #metrics
#Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

ps = PorterStemmer()
nltk.download('all')

pd.set_option('display.max_colwidth',100)
data = pd.read_csv(r"data_emotion.csv")
data.head()

data.info()


plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=data)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def remove_punct(text):
    txt_nopunct="".join([char for char in text if char not in string.punctuation])
    text="".join([re.sub('@([a-z]+)', '', text)]) #remove user handels
    return  text


def tokanize(text):
    txt_tokenized=word_tokenize(text)
    return  txt_tokenized


stopwords_En=nltk.corpus.stopwords.words('english')
def NoStopwords(tokenized_txt):
    text=[word for word in tokenized_txt if word not in stopwords_En ]
    return  text


def stemming(tokenized_text):
    text=" ".join([ps.stem(word) for word in tokenized_text])
    return(text)

def lem(stemmed_sentence):     #using spacy with pos tagging
    doc = nlp(stemmed_sentence)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return lemmatized_sentence

def unusual_text(text):
    english_vocab= set(i.lower() for i in nltk.corpus.words.words())
    text_vocab= set(i.lower() for i in text if i.isalpha())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

# Data Cleaning
# remove punctuations
data['body_text_nopunct']=data['content'].apply(lambda x: remove_punct(x.lower()))
#tokenize
data['body_text_tokenize']=data['body_text_nopunct'].apply(lambda x:tokanize(x))
# Stopwords
data['body_text_nostop']=data['body_text_tokenize'].apply(lambda x: NoStopwords(x))
#stemming
data['cleaned_text']=data['body_text_nostop'].apply(lambda x:stemming(x))


unusual=unusual_text(data['cleaned_text'])
print(unusual)

for i in range(0, 20000):
    text = word_tokenize(data['cleaned_text'][i].lower())  # tokenize data to check every worي separtly
    for t in text:
        if t in unusual:
            text.remove(t)  # remove unsual words from the dataset
    data['cleaned_text'][i] = " ".join(text)  # put the new cleaned data in the data_clean

data

#count vetorizer
cv = CountVectorizer()
# load the CSV file into a pandas DataFrame
Xfeatures = data['cleaned_text']
X = cv.fit(Xfeatures)


#TF-IDF vectorizer
tfidf_vec = TfidfVectorizer(analyzer='word')
tfidf_vec_fit = tfidf_vec.fit(data['cleaned_text'])
X_tfidf = tfidf_vec.fit_transform(data['cleaned_text'])
print(X_tfidf.shape)
print(X_tfidf)


#split_data
ylables=data['sentiment']
X_train,X_test,y_train,y_test=train_test_split(X_tfidf,ylables,test_size=0.3,random_state=42)#randam state to get same shuffle

#Naive Classifier 0.662
em_model = MultinomialNB().fit(X_train, y_train)
pred_test_MNB = em_model.predict(X_test)
precision = precision_score(y_test, pred_test_MNB,average='weighted',zero_division=1)
recall = recall_score(y_test, pred_test_MNB,average='weighted')
accuracy = accuracy_score(y_test, pred_test_MNB)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))


#Tree 0.81
em_model = tree.DecisionTreeClassifier().fit(X_train, y_train)
pred_test_MNB = em_model.predict (X_test)
precision = precision_score(y_test, pred_test_MNB,average='weighted')
recall = recall_score(y_test, pred_test_MNB,average='weighted')
accuracy = accuracy_score(y_test, pred_test_MNB)#806
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))

#SVC 0.82
em_model = SVC().fit(X_train, y_train)
pred_test_MNB = em_model.predict(X_test)
precision = precision_score(y_test, pred_test_MNB,average='weighted')
recall = recall_score(y_test, pred_test_MNB,average='weighted')#823
accuracy = accuracy_score(y_test, pred_test_MNB)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))

#LinearSVC 0.86 (Best Accuracy)
em_model_lin = LinearSVC().fit(X_train, y_train)      #training model on data
y_pred = em_model_lin.predict(X_test)          #
precision = precision_score(y_test, y_pred,average='weighted')
recall = recall_score(y_test, y_pred,average='weighted')#823
accuracy = accuracy_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))
print(confusion_matrix(y_test,y_pred))
print (classification_report(y_test, y_pred))


#code to test model initially
text = 'he fell in love'
X = tfidf_vec_fit.transform([text])
pred = em_model_lin.predict(X)
if pred[0]=='love':
    print('love')
else:
    print('normal')


#function to test model
def predict_emotion(sample_text,model):
    vect=tfidf_vec_fit.transform([sample_text])
    prediction=model.predict(vect)
    print(prediction[0])

t = 'he is sobbing'
predict_emotion(t,em_model_lin)


with open('tfidf_vec_fit.pickle', 'wb') as handle:
    pickle.dump(tfidf_vec_fit,handle)
# save the model to disk
filename = 'linear.sav'
pickle.dump(em_model_lin, open(filename, 'wb'))

with open('tfidf_vec_fit.pickle', 'rb') as handle:
    tfidf_vec_fit_loaded = pickle.load(handle)    #store the loaded-vectorizer to a varable to use it
with open('linear.sav', 'rb') as handle:
    em_model_lin_loaded = pickle.load(handle)     #store the loaded-model to a varable to use it


predict_emotion(t,em_model_lin_loaded)  #test loaded-model


