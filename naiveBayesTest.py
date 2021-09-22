import unittest
from naiveBayes import NaiveBayes
import nltk


class NaiveBayesTest(unittest.TestCase):

    def setUp(self):
        self.naive = NaiveBayes()
        neg_data = self.naive.tokenize_files(self.naive.get_files_from_dir("./data/neg"), "data/neg")
        pos_data = self.naive.tokenize_files(self.naive.get_files_from_dir("./data/pos"), "data/pos")
        self.naive.training_positive_data, self.naive.testing_positive_data = self.naive.split_data(pos_data)
        self.naive.training_negative_data, self.naive.testing_negative_data = self.naive.split_data(neg_data)
        self.naive.train()

    def test_identifying_positive_data(self):
        total_num_events = 0
        total_false_positive = 0
        total_true_positive = 0
        # all_data = self.naive.tokenize_files(self.naive.get_files_from_dir("./data/pos"), "./data/pos")
        # self.naive.split_data()
        for pos_review in self.naive.testing_positive_data:
            decision, num = self.naive.predict_sentiment(pos_review)
            if decision == "positive":
                total_true_positive += 1
            if decision == "negative":
                total_false_positive += 1
            total_num_events += 1
        print("total num events", total_num_events)
        print("total true positive", total_true_positive)
        print("total false events", total_false_positive)
        print("total % detected succesfully", total_true_positive / total_num_events)

    def test_identifying_positive_data_with_pos(self):
        self.naive.train_pos()
        total_num_events = 0
        total_false_positive = 0
        total_true_positive = 0
        raw_pos_reviews = self.naive.get_raw_text_from_files(self.naive.get_files_from_dir("./data/pos"), "./data/pos")
        for pos_review in raw_pos_reviews:
            decision, num = self.naive.predict_sentiment_pos_tag(pos_review)
            if decision == "positive":
                total_true_positive += 1
            if decision == "negative":
                total_false_positive += 1
            total_num_events += 1
        print("total num events", total_num_events)
        print("total true positive", total_true_positive)
        print("total false events", total_false_positive)
        print("total % detected succesfully", total_true_positive / total_num_events)

        # self.assertEqual("positive", decision, "the decision for this review should be positive")

    def test_identifying_neg_data(self):
        tokenized_neg_reviews = self.naive.tokenize_files(self.naive.get_files_from_dir("./data/neg"), "./data/neg")
        for neg_review in tokenized_neg_reviews:
            decision, num = self.naive.predictSentiment(neg_review)
            self.assertEqual("negative", decision, "the decision for this review should be negative")


if __name__ == '__main__':
    unittest.main()
