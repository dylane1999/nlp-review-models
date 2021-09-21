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
        self.total_num_pos_words = 0
        self.total_num_neg_words = 0
        self.positive_probabilities = {}
        self.positive_occurrences = {}
        self.negative_probabilities = {}
        self.negative_occurrences = {}

    # get the list of files from the given directory
    def get_files_from_dir(self, directory: str):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    # perform the needed text processing on the given text and export as a tokenized array
    def process_string(self, text: str) -> str:
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
        self.positive_occurrences, self.total_num_pos_words = self.get_word_occurrences(pos_data)
        self.negative_occurrences,  self.total_num_neg_words = self.get_word_occurrences(neg_data)
        self.positive_probabilities = self.get_word_probability(self.positive_occurrences, self.total_num_pos_words)
        self.negative_probabilities = self.get_word_probability(self.negative_occurrences, self.total_num_neg_words)

    def calculate_naive_probaility(self, occurrences: dict, total_num_words: int, input: list[str]):
        #the probability of a word being positie and the prob that each word in the input is pos
        # word occurance + smoothening_factor / total num pos words
        probability = 1/2  # start by setting to probability of negative or possirive
        for word in input:
            if word not in occurrences:
                numerator = 1
            else:
                numerator = occurrences[word] + 1
            probability *= (numerator/total_num_words)
        return probability


    def predict_sentiment_one_gram(self, tokenizedInput: list[str])-> tuple[str, int]:
        pos_prob = self.calculate_naive_probaility(self.positive_occurrences, self.total_num_pos_words, tokenizedInput)
        neg_prob = self.calculate_naive_probaility(self.negative_occurrences, self.total_num_neg_words, tokenizedInput)
        if pos_prob > neg_prob:
            return ["positive", pos_prob]
        return ["negative", neg_prob]



def main():
    # nltk.download("punkt")
    # nltk.download("stopwords")
    naive = NaiveBayes()
    naive.train()
    processedInput = naive.process_string("I purchased this unit due to frequent blackouts in my area and 2 power supplies going bad.  It will run my cable modem, router, PC, and LCD monitor for 5 minutes.  This is more than enough time to save work and shut down.   Equally important, I know that my electronics are receiving clean power. I feel that this investment is minor compared to the loss of valuable data or the failure of equipment due to a power spike or an irregular power supply. As always, Amazon had it to me in &lt;2 business days")
    decison, num = naive.predictSentiment(processedInput)
    return

if __name__ == '__main__':
    main()
