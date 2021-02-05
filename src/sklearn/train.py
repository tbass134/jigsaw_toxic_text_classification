import argparse
from os import pipe
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multioutput import MultiOutputClassifier


import numpy as np
import string
import re
import joblib


from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

import string
import re

class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **tokenizer_params):
        self.stopwords = set(stopwords.words("english"))
        self.punks = string.punctuation
        self.stemmer = PorterStemmer()
        self.lemmer = WordNetLemmatizer()

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        transforms = [
            # self.to_stem,
            self.remove_newline,
            self.remove_punkts,
            self.remove_stopwords,
            self.remove_urls,
            self.to_lower
        ]
        text = X
        for transform in transforms:
            text = transform(text)
        return text

    def to_stem(self, text):
        print()
        return [" ".join([self.lemmer.lemmatize(word) for t in text for word in t.split(" ") ])]

    def remove_stopwords(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.punks]) for t in text]

    def remove_punkts(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.stopwords]) for t in text]
   
    def to_lower(self, text):
        return [t.lower() for t in text]
      
    def remove_urls(self, text):
        URL = "http://\S+|https://\S+"
        return [re.sub(URL,"",t)  for t in text]

    def remove_newline(self,text):
        return [' '.join(t.splitlines())  for t in text]



class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()



def run(args):
    print(f'debug: {args.debug}')
    print(f'save_model: {args.save_model}')

    train_df = pd.read_csv("data/train.csv")

    pipeline = Pipeline([
        ("tokenizer", TokenizerTransformer()),
        ("vectorizer", CountVectorizer(max_features=5000)),
        ("tfidf", TfidfTransformer()), 
        ("clf", MultiOutputClassifier(MultinomialNB()))
    ])
  
    target_cols = filter(lambda x: x != "id" and x != "comment_text", train_df.columns.to_list()) 
    X = train_df["comment_text"]
    y = train_df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state = 42)

    model = pipeline.fit(X_train, y_train)
    print("fitted...")
    y_tests = y_test.to_numpy()
    y_preds = np.transpose(np.array(model.predict_proba(X_test))[:,:,1])

    aucs = []
    #Calculate the ROC-AUC for each of the target column
    for col in range(y_tests.shape[1]):
        auc_score = roc_auc_score(y_tests[:,col],y_preds[:,col])
        print("col",col)
        print("auc_score",auc_score)
        aucs.append(auc_score)
    mean_auc = np.mean(aucs)

    if args.save_model == True:
        joblib.dump(model, args.model_dir+"model.pkl")

    print(mean_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug', default=False,type=bool)
    parser.add_argument('--save_model', default=False,type=bool)
    parser.add_argument('--model_dir', default="/artifacts",type=str)

    args = parser.parse_args()
    run(args)
