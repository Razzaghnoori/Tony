# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from math import log
from collections import defaultdict
import pickle
import dill
import os
import gensim.models.phrases
import re

class word2vec(Word2Vec):
    def __init__(self, *args, **kwargs):
        super(word2vec, self).__init__(*args, **kwargs)
        self.vocab_size = len(self.wv.vocab.keys())
        self.load_embeddings()
        self.save('word2vec/model')
        self.save_word2vec_format('word2vec/cmodel')
        self.dic = dict()
        for i in range(self.vocab_size):
            self.dic[self.index2word[i]] = i

    def nearest(self, word):
        word = word.decode('utf-8')
        most_similar = self.most_similar(word)
        print word
        for w, p in most_similar:
            print w, p
        print '======================='

    def word2index(self, word):
        if isinstance(word, str):
            word = word.decode('utf-8')
        return self.dic.get(word)


class tfidf(defaultdict):
    def __init__(self, use_bigram_transformer=False):
        super(tfidf, self).__init__(lambda: 0)
        self.use_bigram_transformer = use_bigram_transformer

    def generate(self, model, dirname):
        # sentences = SentenceGrabber('docs/clean_XMLs/bigFiles')
        # bigram_phrases = gensim.models.phrases.Phrases(sentences, min_count= 50, threshold= 50, delimiter= "‌")
        # bigram_transformer = gensim.models.phrases.Phraser(bigram_phrases)
        # bigram_transformer.save('Phrases')
        bigram_transformer = gensim.models.phrases.Phraser.load('Phrases')

        tf = defaultdict(lambda: 0)
        idf = defaultdict(lambda: 0)
        N = len(os.listdir(dirname))  # number of documents
        sentences = Fetch('Files/')

        for file_name in os.listdir(dirname):
            occurs = dict()
            for line in open(os.path.join(dirname, file_name)).readlines():
                if self.use_bigram_transformer:
                    words = bigram_transformer[line.split()]
                else:
                    words = line.split()

                for word in words:
                    occurs[word] = 1
                    tf[word] += 1
            for word in occurs.keys():
                idf[word] += 1

        # Normalizing idf
        for word in idf.keys():
            idf[word] = log(1 + (N / float(idf[word])))
        # max_tf = float(max(tf.values()))
        for word in tf.keys():
            tf[word] = 1 + log(tf[word])

        for word in tf.keys():
            self[word] = tf[word] * idf[word]

        minimum = min(self.values())
        for word in self.keys():
            self[word] -= minimum
            if self[word] < 0.5:
                del self[word]

    def save(self):
        with open('tfidf.pkl', 'wb') as f:
            dill.dump(self, f)

    def load(self):
        with open('tfidf.pkl', 'rb') as f:
            return dill.load(f)


class Fetch(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.char_regex = re.compile("[\w.']+")

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = line.lower()
                line = self.char_regex.findall(line)
                yield line
