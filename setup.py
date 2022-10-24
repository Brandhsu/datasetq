import os
from setuptools import setup, find_packages


with open(f"{os.getcwd()}/README.md", "r") as f:
    long_description = f.read()

packages = find_packages(include=["datasetq"])
install_requires = [
    lib.strip() for lib in open(f"{os.getcwd()}/requirements.txt").readlines()
]

version = {}
with open(f"{os.getcwd()}/datasetq/version.py", "r") as f:
    exec(f.read(), version)

setup(
    name="datasetq",
    version=version["__version__"],
    license="MIT",
    author="Brandhsu",
    author_email="brandondhsu@gmail.com",
    url="https://github.com/Brandhsu/datasetq",
    description="A heap queue dataset sampler for loss-based priority sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "dataset",
        "heap",
        "data structure",
        "priority queue",
        "active learning",
        "machine learning",
    ],
    packages=packages,
    python_requires=">=3.6",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
