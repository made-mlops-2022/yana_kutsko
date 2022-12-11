import logging
import os
import shutil
import unittest
from ml_project.src.model_train_utility import train_model
from ml_project.test_data.synthesize_test_data import synthesize_test_data


class TestTrainModelUtility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synthesized_data_path = synthesize_test_data('./ml_project/data/heart_cleveland.csv')

    @classmethod
    def tearDownClass(cls):
        os.remove('ml-ops.log')
        shutil.rmtree('./out')
        shutil.rmtree('./models')

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_model_knn(self):
        config_path = './ml_project/tests/test_configs/knn.yaml'
        train_model(config_path)

        self.assertTrue(os.path.exists('./models/knn.pkl'))

    def test_train_model_logreg(self):
        config_path = './ml_project/tests/test_configs/logreg.yaml'
        train_model(config_path)

        self.assertTrue(os.path.exists('./models/logreg.pkl'))

    def test_train_model_unknown(self):
        config_path = './ml_project/tests/test_configs/unknown_model.yaml'
        with self.assertRaises(ValueError):
            train_model(config_path)

        self.assertFalse(os.path.exists('./models/unknown_model.pkl'))


if __name__ == '__main__':
    unittest.main()




