import unittest
from naiveBayes import NaiveBayes


class NaiveBayesTest(unittest.TestCase):
    def __init__(self):
        self.naive = NaiveBayes()
        self.naive.train()

    def test_identifying_pos_data(self):
        tokenized_pos_reviews = self.naive.tokenize_files(self.get_files_from_dir("./data/pos"))
        for pos_review in tokenized_pos_reviews:
            decision, num = self.naive.predictSentiment(pos_review)
            self.assertEqual("positive", decision, "the decision for this review should be positive")

    def test_identifying_neg_data(self):
        tokenized_neg_reviews = self.naive.tokenize_files(self.get_files_from_dir("./data/neg"))
        for neg_review in tokenized_neg_reviews:
            decision, num = self.naive.predictSentiment(neg_review)
            self.assertEqual("negative", decision, "the decision for this review should be negative")

if __name__ == '__main__':
    unittest.main()
