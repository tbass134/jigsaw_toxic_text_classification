import torch
import argparse
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from models import *

class TextDataset():
    def __init__(self, text, targets, preprocess=True):
        super().__init__()
        self.text = text
        self.target = targets
        self.textProcessor = TextProccessor()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.text)

    def __getItem__(self,idx):
        text = self.text[idx]
        if self.preprocess:
            text = self.textProcessor.transform(text)

        return text


class TextProccessor():

    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.punks = string.punctuation
        self.words = []
        self.vocab2index = {"":0, "UNK":1}

    def transform(self, text):
        for transform in [self.tokenize,self.remove_punkts,self.remove_stopwords,self.remove_urls, self.remove_newline,self.to_lower]:
            text = transform(text)
        print("text",text)
        return text
        
    def create_vocab(self,text):
        for sent in text:
            for word in sent.split(" "):
                self.vocab2index[word] = len(self.words)
                self.words.append(word)

    def encode_sentence(self, sentence, transform=False,size=100):
        if transform:
            sentence = self.transform(sentence)
            
        encoded = np.zeros(size, dtype=int)
        encoding = np.array([self.vocab2index.get(word, self.vocab2index["UNK"]) for word in sentence])
        length = min(size, len(encoding))
        encoded[:length] = encoding[:length]
        return encoded,length
    

    def remove_stopwords(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.stopwords]) for t in text]

    def remove_punkts(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.punks]) for t in text]
   
    def to_lower(self, text):
        return [t.lower() for t in text]

    def tokenize(self, text):
        return [" ".join(word_tokenize(t)) for t in text]

    def remove_urls(self, text):
        URL = "http://\S+|https://\S+"
        return [re.sub(URL,"",t)  for t in text]

    def remove_newline(self,text):
        return [t.replace("\n","") for t in text]


def train(args):
    train_df = pd.read_csv("data/train.csv", nrows=10)
    #using 1 category for now

    train_df = train_df[["comment_text","toxic"]]
    train, test = train_test_split(train_df, test_size=0.3, random_state = 42)
    train_ds = TextDataset(train["comment_text"].values, train["toxic"].values)
    train_dl = DataLoad

    
    # # test_df = pd.read_csv("data/test.csv")
    # p = TextProccessor()
    # #create the vocab for the text
    # text = p.transform(train_df["comment_text"])
    # p.create_vocab(text)
    # print(p.vocab2index)
    # print(p.words)

    # encodings = []
    # for comment in text:
    #     encodings.append(p.encode_sentence(comment))

    # print(encodings.shape)

    # # p.create_vocab(train_df["comment_text"])
    # # train_df['encoded'] = train_df['comment_text'].apply(lambda x: np.array(p.encode_sentence(x)))
    # # print(train_df.head())
    

    # # print(texts)
    # # texts = []
    # # for text in train_df["comment_text"][:10]:
    # #     texts.append(text)
    # # print(texts)

    

    



    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', default=False,type=bool)
    parser.add_argument('--save_model', default=False,type=bool)
    args = parser.parse_args()
    train(args)