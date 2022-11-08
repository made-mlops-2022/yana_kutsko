import os
import logging
import shutil
import unittest
from synthesize_test_data import synthesize_test_data
from model_train_utility import train_model
from model_predict_utility import predict_model


class TestTrainModelUtility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synthesized_data_path = synthesize_test_data('../data/heart_cleveland.csv')

    @classmethod
    def tearDownClass(cls):
        os.remove('ml-ops.log')
        shutil.rmtree('./out')
        shutil.rmtree('./models')
        shutil.rmtree('./predictions')

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_predict_model_knn(self):
        config_path = './test_configs/knn.yaml'
        train_model(config_path)

        model_path = './models/knn.pkl'
        data_csv_path = './out/random_mode/synthetic_data.csv'
        output_path = './predictions/predictions_knn.csv'
        predict_model(model_path, data_csv_path, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertNotEqual(os.stat(output_path).st_size, 0)

    def test_predict_model_logreg(self):
        config_path = './test_configs/logreg.yaml'
        train_model(config_path)

        model_path = './models/logreg.pkl'
        data_csv_path = './out/random_mode/synthetic_data.csv'
        output_path = './predictions/predictions_logreg.csv'
        predict_model(model_path, data_csv_path, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertNotEqual(os.stat(output_path).st_size, 0)

    def test_predict_model_unknown(self):
        model_path = './models/unknown_model.pkl'
        data_csv_path = './out/random_mode/synthetic_data.csv'
        output_path = './predictions/predictions_unknown_model.csv'

        with self.assertRaises(FileNotFoundError):
            predict_model(model_path, data_csv_path, output_path)






