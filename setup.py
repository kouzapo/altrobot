#!/usr/bin/env python3

import setuptools

with open('README.md', 'r', encoding = 'utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = 'altrobot',
    version = '0.0.1',
    author = 'Apostolos Anastasios Kouzoukos',
    author_email = 'kouzoukos97@gmail.com',
    description = 'A backtesting library for algorithmic trading, based on neural networks',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/kouzapo/altrobot',
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires = '>=3.6',
)
