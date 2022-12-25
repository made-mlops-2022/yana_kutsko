from setuptools import setup, find_packages


setup(
    name='ml_ops',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
            "console_scripts": [
                "train_model = ml_project.model_train_utility:train_model_command",
                "predict_model = ml_project.model_predict_utility:predict_model_command"
            ]
        },
)