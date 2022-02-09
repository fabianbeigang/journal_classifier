#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:32:45 2022

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
        self.model = None
        pass

    def fit(self, train, test, y=None):
        # Initialize model
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=3, epochs=40)
        # Build model vocabulary
        corp = generate_corpus(train)
        self.model.build_vocab(corp)
        # Build model 
        self.model.train(corp, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, x_dataset):
        if isinstance(x_dataset,str):
            return pd.DataFrame([vector_represent(x_dataset,self.model)])
        else:
            return pd.DataFrame([vector_represent(x,self.model) for x in x_dataset])


## Functions

# Define function for text preprocessing and inferring its vector (model should be)
def vector_represent(text,model):
    
    # Remove stopwords
    p_text = remove_stopwords(text)
    # Preprocess
    p_text = gensim.utils.simple_preprocess(p_text, max_len=30)
    # Get vector
    vector = model.infer_vector(p_text)
    #tagged_text = gensim.models.doc2vec.TaggedDocument(p_text, [i])
    return vector

 

# Function to preprocess a text document and turn it into a tagged document
def generate_corpus(text_docs):
    corpus = []
    for i, text_doc in enumerate(text_docs):
        prep_text = gensim.utils.simple_preprocess(text_doc, max_len=30)
        tagged_doc = gensim.models.doc2vec.TaggedDocument(prep_text, [i])
        corpus.append(tagged_doc)
    return corpus

# Extract information about textmodel from pipeline string
def get_textmodel(pipeline):
    if "tfidfvectorizer" in pipeline:
        return "TFIDF"
    else:
        return "d2v"

# Extract information about classifier from pipeline string
def get_classifier(pipeline):
    if "logisticregression" in pipeline:
        return "LogReg"
    else:
        return "XGB"

# Extract information about hyperparameters from pipeline string
def get_hyperparams(pipeline):
    if "logisticregression" in pipeline:
        return pipeline.split("LogisticRegression(")[1].replace(")","").replace("]","").replace("\n","").replace(" ","")
    else:
        return pipeline.split("XGBClassifier(")[1].replace(")","").replace("]","").replace("\n","").replace(" ","")

