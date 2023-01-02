# TitanicClassifier.jl

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5/blob/master/LICENSE)

TitanicClassifier is a package which applies SVM algorithm to the problem of classification of the survivors of the [Titanic](https://www.kaggle.com/c/titanic).

## Installation

The package is not registered and this can be installed in the following way

```julia
(@v1.8) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5
```

## Detailed information

The main goal of the package is to provide a simple interface for choosing hyperparameters and training of the SVM algorithm on the Titanic dataset and subsequent classification of unlabeled data.

The package provides several general methods for data preprocessing (see [data_preparation.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5/blob/master/src/data_preparation.jl)) and training of the SVM algorithm (see [svm.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5/blob/master/src/svm.jl)), and therefore its usage is not restricted exclusively to the aforementioned Titanic dataset. It does however also contain methods made specifically for the structure of the Titanic dataset (see TitanicClassifier.jl).

## Examples

The module also contains two jupyter notebooks. One provides further information about feature engineering of the Titanic dataset (see [features.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5/blob/master/examples/features.ipynb)). The other contains exemplary usage of methods included in the package (see [usage.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cechdav5/blob/master/examples/usage.ipynb)).
