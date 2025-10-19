from setuptools import setup, find_packages

setup(
    name="lunar_mnistnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "mlflow>=2.10",
        "pytest"
    ],
)