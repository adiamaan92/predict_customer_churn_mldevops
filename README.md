<div align='center'>

# Predict Customer Churn - MLDev Ops

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/adiamaan92/modi-speech-scrapper/blob/master/MIT-LICENSE.txt)

</div>

![project1](/cover.png)

This project predicts the customer churn in a bank. This project is part of ML DevOps Engineer Nanodegree from Udacity. The idea is to **structure a simple ML project using the best practices borrowed from software engineering**

## 💡 Project Description

The primary task of this project is to follow the software engineering processes and less emphasis on the prediction itself. The task includes,
1. pylint and autopep-8 for adhering to pep-8 standards
2. Modularization and documentation of functions
3. Unit test and logging


## 🧰 Setting up environment

Poetry is used as the venv manager. The environment can be setup by following the steps,

1. Install poetry  
`pip install poetry`

2. Install dependencies (including dev)  
`poetry install`

## 🏃 Running Files

The main churn prediction code can be run as,  
`python churn_library.py`, which generates EDA plots, result plots and stores the models

The testing script can be run as,
`python churn_script_logging_and_tests.py`, which runs the tests and logs the results

## 🎯 Pylint scores

1. `pylint churn_library.py` yields a score of 7.17/10
2. `pylint churn_script_logging_and_tests.py` yields a score of 7.14/10


## ✨ Additional improvements

On top of the best practices required by the project the following improvements are added as well,

- Poetry for environment setup and reproducibility
- Add typing information to all the functions
- Mypy for static type checking
- Flake8 for linting warnings
- isort for sorting inputs
- Pre-commit to check for flake8, isort, pytest, pylint and mypy warnings
