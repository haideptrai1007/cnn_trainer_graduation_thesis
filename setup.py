from setuptools import setup

setup(
    name="trainer",
    version="0.1.0",
    py_modules=["trainer"],
    install_requires=["torch", "numpy", "scikit-learn", "matplotlib", "seaborn"],
)