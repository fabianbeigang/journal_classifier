# Explainable Journal Recommendations
In this project, I built a classifier which recommends the best journal on the basis of the abstract and title of a philosophy paper. Journal recommendations are explained using the LIME framework.

* [The data extraction and preparation notebook](https://github.com/fabianbeigang/journal_classifier/blob/main/journal_dataset_generation.ipynb)
* [The modelling notebook](https://github.com/fabianbeigang/journal_classifier/blob/main/journal_modelling.ipynb)
* [The module containing specific functions](https://github.com/fabianbeigang/journal_classifier/blob/main/functions.py)

The project showcases the following skills:

* Text modelling (TF-IDF, doc2vec) using *gensim* and *sklearn*
* Building a machine learning pipeline using *sklearn* and *xgboost*
* Resolving class imbalance problems using *SMOTE*
* Making classifiers explainable using *LIME*
