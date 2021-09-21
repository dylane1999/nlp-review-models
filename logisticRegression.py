from os import listdir
from os.path import isfile, join

import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import sklearn
import pandas
import os
from os import listdir
from os.path import isfile, join

class log:

    def predict_sentiment_pos_tag(self, raw_text:str) -> tuple[str, int]:
        pos = nltk.pos_tag(word_tokenize(raw_text))
        pos_prediction = self.calculate_naive_probaility(self.positive_occurrences, len(self.positive_occurrences), pos)
        neg_prediction = self.calculate_naive_probaility(self.negative_occurrences, len(self.negative_occurrences), pos)
        if pos_prediction > neg_prediction:
            return ["positive", pos_prediction]
        return ["negative", neg_prediction]


    def train_pos(self):
        negative_data = self.get_raw_text_from_files(self.get_files_from_dir("./data/neg"), "./data/neg")
        positive_data = self.get_raw_text_from_files(self.get_files_from_dir("./data/pos"), "./data/pos")
        negative_pos = []
        positive_pos = []
        for raw_text in negative_data:
            pos = nltk.pos_tag(nltk.word_tokenize(raw_text))
            negative_pos.append(pos)
        for raw_text in positive_data:
            pos = nltk.pos_tag(nltk.word_tokenize(raw_text))
            positive_pos.append(pos)
        self.positive_occurrences, self.total_num_pos_words = self.get_word_occurrences(positive_pos)
        self.negative_occurrences, self.total_num_neg_words = self.get_word_occurrences(negative_pos)
        self.positive_probabilities = self.get_word_probability(self.positive_occurrences, len(self.positive_occurrences))
        self.negative_probabilities = self.get_word_probability(self.negative_occurrences, len(self.negative_occurrences))
