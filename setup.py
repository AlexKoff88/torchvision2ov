from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="tv2ov",
    version="0.1",
    install_requires=required,
    packages=find_packages(),
)