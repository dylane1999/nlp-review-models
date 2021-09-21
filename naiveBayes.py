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



    # get the list of files from the given directory
    def get_files_from_dir(self, directory: str):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    # perform the needed text processing on the given text and export as a tokenized array
    def process_string(self, text: str) -> str:
        englishStopwords = stopwords.words("english")  # non-neccesary words
        text = text.lower()  # case folding
        # remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])
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
        self.positive_occurrences, self.total_num_pos_words = self.get_word_occurrences(self.training_positive_data)
        self.negative_occurrences, self.total_num_neg_words = self.get_word_occurrences(self.training_negative_data)
        self.positive_probabilities = self.get_word_probability(self.positive_occurrences, self.total_num_pos_words)
        self.negative_probabilities = self.get_word_probability(self.negative_occurrences, self.total_num_neg_words)

    def calculate_naive_probaility(self, occurrences: dict, total_num_words: int, input: list[str]):
        # the probability of a word being positie and the prob that each word in the input is pos
        # word occurance + smoothening_factor / total num pos words
        probability = 1 / 2  # start by setting to probability of negative or possirive
        for word in input:
            if word not in occurrences:
                numerator = 1
            else:
                numerator = occurrences[word] + 1
            probability *= (numerator / total_num_words)
        return probability

    def predict_sentiment_one_gram(self, tokenizedInput: list[str]) -> tuple[str, int]:
        pos_prob = self.calculate_naive_probaility(self.positive_occurrences, self.total_num_pos_words, tokenizedInput)
        neg_prob = self.calculate_naive_probaility(self.negative_occurrences, self.total_num_neg_words, tokenizedInput)
        if pos_prob > neg_prob:
            return ["positive", pos_prob]
        return ["negative", neg_prob]

    def split_data(self, all_data: list[str]) -> tuple[list[str], list[str]]:
        random.shuffle(all_data)
        training_data = all_data[0:len(all_data)//2]
        testing_data = all_data[len(all_data)//2:]
        return [training_data, testing_data]

    def get_num_exclams_in_text(self, raw_text: str) -> int:
        num_exclams = 0
        tokenized = word_tokenize(raw_text)
        for word in tokenized:
            if word == "!":
                num_exclams += 1
        return num_exclams

    def get_num_of_positive_words_in_text(self, raw_text: str) -> int:
        pos_list=set(opinion_lexicon.positive())
        # neg_list=set(opinion_lexicon.negative())
        positive_dict = [self.porter.stem(word) for word in pos_list]
        words_in_text = [self.porter.stem(word) for word in word_tokenize(raw_text)]
        num_positive = 0
        for word in words_in_text:
            if word in positive_dict:
                num_positive += 1
        return num_positive


def main():
    # nltk.download("punkt")
    # nltk.download("stopwords")
    # neg_data = self.tokenize_files(self.get_files_from_dir("./data/neg"), "data/neg")
    # pos_data = self.tokenize_files(self.get_files_from_dir("./data/pos"), "data/pos")
    naive = NaiveBayes()
    naive.train()
    raw_input = "I purchased this unit due to frequent blackouts in my area and 2 power supplies going bad.  It will run my cable modem, router, PC, and LCD monitor for 5 minutes.  This is more than enough time to save work and shut down.   Equally important, I know that my electronics are receiving clean power. I feel that this investment is minor compared to the loss of valuable data or the failure of equipment due to a power spike or an irregular power supply. As always, Amazon had it to me in &lt;2 business days"
    decison, num = naive.predict_sentiment_pos_tag(raw_input)
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