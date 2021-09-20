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


class NaiveBayes:
    def __init__(self):
        self.pos_files = []
        self.neg_files = []
        self.total_num_words = 0
        self.positive_probabilities = {}
        self.positive_occurrences = {}
        self.negative_probabilities = {}
        self.negative_occurrences = {}

    # get the list of files from the given directory
    def get_files_from_dir(self, directory: str):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    # perform the needed text processing on the given text and export as a tokenized array
    def process_string(self, text: str):
        porter = PorterStemmer()  # removes stems from words
        englishStopwords = stopwords.words("english")  # non-neccesary words
        text = text.lower()  # case folding
        # remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])
        words = word_tokenize(text)
        removed = [word for word in words if word not in englishStopwords]
        stemmed = [porter.stem(word) for word in removed]
        return stemmed

    def tokenize_files(self, files: list, dir: str) -> list:
        cleaned_positive_files = []
        for file in files:
            file_path = str.format("{}/{}", dir, file)
            print(file_path)
            with open(file_path) as f:
                file_text_in_lines = f.readlines()
                for line in file_text_in_lines:
                    cleaned_positive_files.append(self.process_string(line))
        return cleaned_positive_files

    def get_word_occurrences(self, tokenized_files: list) -> tuple[dict, int]:
        word_occurrences = {}
        total_num_words = 0
        for file in tokenized_files:
            for word in file:
                total_num_words += 1
                if word not in word_occurrences:
                    word_occurrences[word] = 0
                word_occurrences[word] += 1
        return word_occurrences, total_num_words

    def get_word_probability(self, word_occurrences: dict, total_num_words: int) -> dict:
        word_probabilities = {}
        for word, num_occurrences in word_occurrences.items():
            word_probabilities[word] = num_occurrences / total_num_words
        return word_probabilities

    def train(self):
        neg_data = self.tokenize_files(self.get_files_from_dir("./data/neg"), "data/neg")
        pos_data = self.tokenize_files(self.get_files_from_dir("./data/pos"), "data/pos")
        self.positive_occurrences = self.get_word_occurrences(pos_data)
        self.negative_occurrences = self.get_word_occurrences(neg_data)
        self.positive_probabilities = self.get_word_probability(self.positive_occurrences)
        self.negative_probabilities = self.get_word_probability(self.negative_occurrences)

    def predictSentiment(self, input: str)-> tuple[str, int]:
        return



def main():
    nltk.download("punkt")
    nltk.download("stopwords")
    naive = NaiveBayes()
    naive.train()
    return

main()

