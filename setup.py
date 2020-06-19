from setuptools import setup, find_packages
import os

setup(
    name="torch_yolo",
    packages=find_packages(),
    install_requires=["numpy", "torch", "torchvision", "matplotlib", "terminaltables", "pillow", "tqdm"],
    version="0.1"
)
