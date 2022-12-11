import os
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


def synthesize_test_data(input_data_path='../data/heart_cleveland.csv'):
    mode = 'random_mode'
    description_file = f'./out/{mode}/description.json'
    synthetic_data = f'./out/{mode}/synthetic_data.csv'

    os.makedirs(os.path.dirname(description_file), exist_ok=True)
    os.makedirs(os.path.dirname(synthetic_data), exist_ok=True)

    threshold_value = 20

    num_tuples_to_generate = pd.read_csv(input_data_path).shape[0]

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_random_mode(input_data_path)
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_random_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    return synthetic_data



