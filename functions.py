#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:32:45 2022

This module provides a number of functions for the explainable journal 
recommender project.

@author: fabianbeigang
"""

import gensim
import pandas as pd
from sklearn.base import BaseEstimator
from gensim.parsing.preprocessing import remove_stopwords

## Classes

#SKLearn-style wrapper to generate a doc2vec document model
class DocVectorizer(BaseEstimator):
    
    
    def __init__(self):
        """
        Creates the DocVectorizer object.
        """
        self.model = None
        pass

    def fit(self, train, test, y=None):
        """

        Parameters
        ----------
        train : list
            List of text documents.
        test : list
            List of text documents.
        y : list, optional
            List of labels. The default is None.

        Returns
        -------
        TYPE
            Returns the fitted text model.

        """
        # Initialize model
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=3, epochs=40)
        # Build model vocabulary
        corp = generate_corpus(train)
        self.model.build_vocab(corp)
        # Build model 
        self.model.train(corp, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, x_dataset):
        """
        

        Parameters
        ----------
        x_dataset : list or str
            List of text documents or individual text document.

        Returns
        -------
        DataFrame
            Returns a data frame with the vector representations of the text documents.

        """
        if isinstance(x_dataset,str):
            return pd.DataFrame([vector_represent(x_dataset,self.model)])
        else:
            return pd.DataFrame([vector_represent(x,self.model) for x in x_dataset])


## Functions


def vector_represent(text,model):
    """
    Defines a function for text preprocesing and inferring its vector.

    Parameters
    ----------
    text : str
        Text that is to be represented as a vector.
    model : BaseEstimator
        The text model.

    Returns
    -------
    vector : Series
        Vector representation of text.

    """
    
    # Remove stopwords
    p_text = remove_stopwords(text)
    # Preprocess
    p_text = gensim.utils.simple_preprocess(p_text, max_len=30)
    # Get vector
    vector = model.infer_vector(p_text)
    
    return vector

 
def generate_corpus(text_docs):
    """
    Function to preprocess a text document and turn it into a tagged document.

    Parameters
    ----------
    text_docs : TYPE
        

    Returns
    -------
    corpus : TYPE
        

    """
    corpus = []
    for i, text_doc in enumerate(text_docs):
        prep_text = gensim.utils.simple_preprocess(text_doc, max_len=30)
        tagged_doc = gensim.models.doc2vec.TaggedDocument(prep_text, [i])
        corpus.append(tagged_doc)
    return corpus

# Extract information about textmodel from pipeline string
def get_textmodel(pipeline):
    """
    Function to extract information about textmodel from pipeline string.

    Parameters
    ----------
    pipeline : TYPE

    Returns
    -------
    str

    """
    if "tfidfvectorizer" in pipeline:
        return "TFIDF"
    else:
        return "d2v"


def get_classifier(pipeline):
    """
    Function to extract information about classifier from pipeline string.

    Parameters
    ----------
    pipeline : TYPE

    Returns
    -------
    str

    """
    if "logisticregression" in pipeline:
        return "LogReg"
    else:
        return "XGB"


def get_hyperparams(pipeline):
    """
    Function to extract information about hyperparameters from pipeline string.

    Parameters
    ----------
    pipeline : TYPE

    Returns
    -------
    str

    """
    if "logisticregression" in pipeline:
        return pipeline.split("LogisticRegression(")[1].replace(")","").replace("]","").replace("\n","").replace(" ","")
    else:
        return pipeline.split("XGBClassifier(")[1].replace(")","").replace("]","").replace("\n","").replace(" ","")

