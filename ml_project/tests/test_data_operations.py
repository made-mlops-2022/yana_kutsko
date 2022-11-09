import logging
import os
import shutil
import unittest
from ml_project.src.data.dataset_operations import DatasetOperations
from ml_project.test_data.synthesize_test_data import synthesize_test_data


class TestDataOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synthesized_data_path = synthesize_test_data('./ml_project/data/heart_cleveland.csv')
        cls.dataset_operations = DatasetOperations()

    @classmethod
    def tearDownClass(cls):
        os.remove('ml-ops.log')
        shutil.rmtree('./out')

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_read_dataset(self):
        df = self.dataset_operations.read_dataset(self.synthesized_data_path)
        self.assertIsNotNone(df)
        self.assertNotEqual(df.shape, (0, 0))

    def test_split_to_features_and_target(self):
        df = self.dataset_operations.read_dataset(self.synthesized_data_path)
        features, target = self.dataset_operations.split_to_features_and_target(df)

        df_shape = df.shape
        features_shape = features.shape
        target_shape = target.shape

        self.assertNotEqual(df_shape, (0, 0))
        self.assertNotEqual(features_shape, (0, 0))
        self.assertNotEqual(target_shape, (0, 0))

        self.assertEqual(features_shape, (df_shape[0], df_shape[1] - 1))
        self.assertEqual(target_shape, (df_shape[0],))

    def test_split_to_train_and_test(self):
        df = self.dataset_operations.read_dataset(self.synthesized_data_path)
        features, target = self.dataset_operations.split_to_features_and_target(df)

        df_shape = df.shape
        features_shape = features.shape
        target_shape = target.shape

        self.assertNotEqual(df_shape, (0, 0))
        self.assertNotEqual(features_shape, (0, 0))
        self.assertNotEqual(target_shape, (0, 0))

        test_size = 0.33
        random_state = 42

        X_train, X_test, y_train, y_test = self.dataset_operations.split_to_train_and_test(features, target,
                                                                                           test_size, random_state)

        self.assertNotEqual(X_train.shape, (0, 0))
        self.assertNotEqual(X_test.shape, (0, 0))
        self.assertNotEqual(y_train.shape, (0, 0))
        self.assertNotEqual(y_test.shape, (0, 0))

        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertNotEqual(y_train.shape, y_test.shape)


if __name__ == '__main__':
    unittest.main()







