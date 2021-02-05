
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string
import re

class Preprocess():
    def __init__(self):
        super(Preprocess).__init__()
        self.stopwords = set(stopwords.words("english"))
        self.punks = string.punctuation

    def transform(self, text):
        for transforms in [self.tokenize, self.remove_punkts, self.remove_stopwords, self.remove_urls, self.to_lower]:
            text = transforms(text)

        # text = self.tokenize(text)
        # text = self.remove_punkts(text)
        # text = self.remove_stopwords(text)
        # text = self.remove_urls(text)
        # text = self.to_lower(text)
        return text

    def remove_stopwords(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.stopwords]) for t in text]

    def remove_punkts(self, text):
        return [" ".join([word for word in t.split(" ") if word not in self.punks]) for t in text]
   
    def to_lower(self, text):
        return [t.lower() for t in text]

    def tokenize(self, text):
        return [ " ".join(word_tokenize(t)) for t in text]

    def remove_urls(self, text):
        URL = "http://\S+|https://\S+"
        return [re.sub(URL,"",t)  for t in text]