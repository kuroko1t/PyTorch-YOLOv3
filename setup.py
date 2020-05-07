from setuptools import setup, find_packages
import os

setup(
    name="torchyolo",
    packages=find_packages(),
    install_requires=["numpy", "torch", "matplotlib", "terminaltables", "pillow", "tqdm"],
    version="0.1"
)
