# -*- coding: utf-8 --*

import utilities
import gensim.models.phrases
import numpy as np
#import tensorflow as tf
from datetime import datetime as dt
from os.path import join
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import confusion_matrix

def train_bigram_transformer(addr='docs/clean_XMLs/bigFiles', min_count=50, threshold=50, save_addr='Phrases'):
    """
    Creates bigram_transformer and saves it

    `min_count` ignore all words and bigrams with total collected count lower than this.

    `threshold` represents a threshold for forming the phrases (higher means
    fewer phrases). A phrase of words `a` and `b` is accepted if
    `(cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold`, where `N` is the
    total vocabulary size.
    """
    
    sentences = utilities.SentenceGrabber(addr)
    bigram_phrases = gensim.models.phrases.Phrases(sentences, min_count, threshold, delimiter= "_")
    bigram_transformer = gensim.models.phrases.Phraser(bigram_phrases)
    bigram_transformer.save(save_addr)

def train_word2vec(addr='docs/clean_XMLs/bigFiles', vector_size=50, use_bigram_transform=False,
                   bigram_transformer_file_addr='Phrases', _window=8, _min_count=10, _workers=4,
                   _iter=5, _null_word=1):
    """
    Train word2vec based on folder of files provided as "addr" and embedding size provided
    as "vector_size"
    """
    
    sentences = utilities.SentenceGrabber(addr)
    if use_bigram_transform:
        bigram_transformer = gensim.models.phrases.Phraser.load(bigram_transformer_file_addr)
        model = Word2Vec(bigram_transformer[sentences], size=vector_size, window=_window, \
                                   min_count=_min_count, workers=_workers, iter=_iter, null_word=_null_word)
    else:
        model = Word2Vec(sentences, size=vector_size, window=_window, \
                                   min_count=_min_count, workers=_workers, iter=_iter, null_word=_null_word)
    return model


