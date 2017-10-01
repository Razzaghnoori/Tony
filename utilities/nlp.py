# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
import pickle
import os
import gensim.models.phrases
from math import log


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


class tfidf(dict):
    def generate(self, model, dirname):
        # sentences = SentenceGrabber('docs/clean_XMLs/bigFiles')
        # bigram_phrases = gensim.models.phrases.Phrases(sentences, min_count= 50, threshold= 50, delimiter= "‌")
        # bigram_transformer = gensim.models.phrases.Phraser(bigram_phrases)
        # bigram_transformer.save('Phrases')
        bigram_transformer = gensim.models.phrases.Phraser.load('Phrases')

        tf = dict()
        idf = dict()
        N = len(os.listdir(dirname))  # number of documents

        for doc_name in os.listdir(dirname):
            occurs = dict()
            for i, line in enumerate(open(os.path.join(dirname, doc_name)).readlines()):
                for word in bigram_transformer[line.split()]:
                    occurs[word] = 1
                    if tf.get(word):
                        tf[word] += 1
                    else:
                        tf[word] = 1
            for word in occurs.keys():
                if idf.get(word):
                    idf[word] += 1
                else:
                    idf[word] = 1

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
            pickle.dump(self, f)

    def load(self):
        with open('tfidf.pkl', 'rb') as f:
            return pickle.load(f)


class Fetch(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = line.decode('utf-8')
                yield line.split()
