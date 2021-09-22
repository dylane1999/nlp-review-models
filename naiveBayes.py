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
import random
from nltk.corpus import opinion_lexicon

nltk.download('opinion_lexicon')


class NaiveBayes:
    def __init__(self):
        self.pos_files = []
        self.neg_files = []
        self.total_num_pos_words = 0
        self.total_num_neg_words = 0
        self.positive_probabilities = {}
        self.positive_occurrences = {}
        self.negative_probabilities = {}
        self.negative_occurrences = {}
        self.training_negative_data = []
        self.training_positive_data = []
        self.testing_negative_data = []
        self.testing_positive_data = []
        self.porter = PorterStemmer()  # removes stems from words
        self.positive_dict = set(opinion_lexicon.positive())
        self.positive_dict_stemmed = [self.porter.stem(word) for word in self.positive_dict]
        self.negative_dict = set(opinion_lexicon.negative())
        self.negative_dict_stemmed = [self.porter.stem(word) for word in self.negative_dict]
        self.punctuation = string.punctuation.replace("!", "")

    # get the list of files from the given directory
    def get_files_from_dir(self, directory: str):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    # perform the needed text processing on the given text and export as a tokenized array
    def process_string(self, text: str) -> str:
        englishStopwords = stopwords.words("english")  # non-neccesary words
        text = text.lower()  # case folding
        # remove punctuation
        text = "".join([char for char in text if char not in self.punctuation])
        words = word_tokenize(text)
        removed = [word for word in words if word not in englishStopwords]
        stemmed = [self.porter.stem(word) for word in removed]
        return stemmed

    def tokenize_files(self, files: list, dir: str) -> list:
        cleaned_positive_files = []
        for file in files:
            file_path = str.format("{}/{}", dir, file)
            with open(file_path) as f:
                file_text_in_lines = f.readlines()
                for line in file_text_in_lines:
                    cleaned_positive_files.append(self.process_string(line))
        return cleaned_positive_files

    def get_raw_text_from_files(self, files: list, dir: str) -> list:
        raw_text = []
        for file in files:
            file_path = str.format("{}/{}", dir, file)
            with open(file_path) as f:
                file_text_in_lines = f.read()
                raw_text.append(file_text_in_lines)
        return raw_text

    def get_word_occurrences(self, tokenized_files: list) -> tuple[dict, int]:
        word_occurrences = {}
        word_occurrences["positive"] = 0
        word_occurrences["negative"] = 0
        total_num_words = 0
        for file in tokenized_files:
            # calc number exclams
            # calc number pos/neg/words
            for word in file:
                if self.is_word_positive(word):
                    word_occurrences["positive"] += 1
                if self.is_word_negative(word):
                    word_occurrences["negative"] += 1
                if word not in word_occurrences:
                    word_occurrences[word] = 0
                word_occurrences[word] += 1
                total_num_words += 1
        return word_occurrences, total_num_words

    def get_word_probability(self, word_occurrences: dict, total_num_words: int) -> dict:
        word_probabilities = {}
        for word, num_occurrences in word_occurrences.items():
            word_probabilities[word] = num_occurrences / total_num_words
        return word_probabilities

    def train(self):
        self.positive_occurrences, self.total_num_pos_words = self.get_word_occurrences(self.training_positive_data)
        self.negative_occurrences, self.total_num_neg_words = self.get_word_occurrences(self.training_negative_data)
        self.positive_probabilities = self.get_word_probability(self.positive_occurrences, self.total_num_pos_words)
        self.negative_probabilities = self.get_word_probability(self.negative_occurrences, self.total_num_neg_words)

    def calculate_naive_probaility(self, occurrences: dict, total_num_words: int, input: list[str]):
        # the probability of a word being positie and the prob that each word in the input is pos
        # word occurance + smoothening_factor / total num pos words
        num_positive_words = 1  # start w/ one for smoothening
        num_negative_words = 1
        num_exclams = 1
        probability = 1 / 2  # start by setting to probability of negative or possirive
        for word in input:
            if self.is_word_positive(word):
                num_positive_words += 1
            if self.is_word_negative(word):
                num_negative_words += 1
            if word == "!":
                num_exclams += 1
            if word not in occurrences:
                numerator = 1
            else:
                numerator = occurrences[word] + 1
            probability *= (numerator / total_num_words)
        probability *= ((num_exclams + occurrences["!"]) / total_num_words)  # add exclamation feature
        probability *= ((num_positive_words + occurrences["positive"]) / total_num_words)  # add positive words feature
        probability *= ((num_negative_words + occurrences["negative"]) / total_num_words)  # add negative words feature
        return probability

    def predict_sentiment(self, tokenizedInput: list[str]) -> tuple[str, int]:
        pos_prob = self.calculate_naive_probaility(self.positive_occurrences, self.total_num_pos_words, tokenizedInput)
        neg_prob = self.calculate_naive_probaility(self.negative_occurrences, self.total_num_neg_words, tokenizedInput)
        if pos_prob > neg_prob:
            return ["positive", pos_prob]
        return ["negative", neg_prob]

    def split_data(self, all_data: list[str]) -> tuple[list[str], list[str]]:
        # random.shuffle(all_data)
        training_data = all_data[0:len(all_data) // 2]
        testing_data = all_data[len(all_data) // 2:]
        return [training_data, testing_data]

    def get_num_exclams_in_text(self, tokenized_input: list[str]) -> int:
        num_exclams = 0
        for word in tokenized_input:
            if word == "!":
                num_exclams += 1
        return num_exclams

    def is_word_positive(self, word: str) -> bool:
        if word in self.positive_dict or word in self.positive_dict_stemmed:
            return True
        return False

    def is_word_negative(self, word: str) -> bool:
        if word in self.negative_dict or word in self.negative_dict_stemmed:
            return True
        return False


def main():
    naive = NaiveBayes()
    # nltk.download("punkt")
    # nltk.download("stopwords")
    neg_data = naive.tokenize_files(naive.get_files_from_dir("./data/neg"), "data/neg")
    pos_data = naive.tokenize_files(naive.get_files_from_dir("./data/pos"), "data/pos")
    naive.training_positive_data, naive.testing_positive_data = [pos_data, pos_data]  # naive.split_data(pos_data)
    naive.training_negative_data, naive.testing_negative_data = [neg_data, neg_data]  # naive.split_data(neg_data)

    naive.train()
    input = naive.process_string(
        "I purchased this unit due to frequent blackouts in my area and 2 power supplies going bad.  It will run my cable modem, router, PC, and LCD monitor for 5 minutes.  This is more")
    decison, num = naive.predict_sentiment(input)
    return


if __name__ == '__main__':
    main()

# 2 gram
# how consistent
# look a prob set indepentdly to make a decison
# laballing words in a sentiment dict
# labelling words in a sentietn a
# use the num of pos words in a sentecnce as a feaure in naive bayes
# feature for number of ! marks

#  bottom line asfjkld
