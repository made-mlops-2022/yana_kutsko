import os
import unittest
from synthesize_test_data import synthesize_test_data
from model_train_utility import train_model


class TestTrainModelUtility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synthesized_data_path = synthesize_test_data('../data/heart_cleveland.csv')

    def test_train_model_knn(self):
        config_path = './test_configs/knn.yaml'
        train_model(config_path)

        self.assertTrue(os.path.exists('./models/knn.pkl'))

    def test_train_model_logreg(self):
        config_path = './test_configs/logreg.yaml'
        train_model(config_path)

        self.assertTrue(os.path.exists('./models/logreg.pkl'))

    def test_train_model_unknown(self):
        config_path = './test_configs/unknown_model.yaml'
        with self.assertRaises(ValueError):
            train_model(config_path)

        self.assertFalse(os.path.exists('./models/unknown_model.pkl'))




