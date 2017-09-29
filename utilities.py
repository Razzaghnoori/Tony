# -*- coding: utf-8 -*-

from codecs import open as uopen
from gensim.models import Word2Vec
from math import log

import re
import collections
import pickle
import os
import gensim.models.phrases
import numpy as np


html_regex = re.compile("<[^>]*>")
non_persian_regex = re.compile(u"[^َُِأاآبپتثجچحخدذرزژسشصضطظعغفقکكگلمنوؤهیيئ‌ء ۰-۹\.]")
puncRegex = re.compile('[؟‌\u200c\(\)\"\'\-\\\u060c\&%#\*\|\u061b«»،]')
paragraphRegex = re.compile('<p>(.*)</p>')
space_regex = re.compile('[\s\t\n\r]{2,}')
dot2enter_regex = re.compile('[\.؟!]+ ')


##################################################
##################################################
#CLASSES
##################################################
##################################################



class SentenceGrabber(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = line.decode('utf-8')
                yield line.split()


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
        #sentences = SentenceGrabber('docs/clean_XMLs/bigFiles')
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

        #Normalizing idf
        for word in idf.keys():
            idf[word] = log(N / float(idf[word]))
        # max_tf = float(max(tf.values()))
        for word in tf.keys():
            tf[word] = log(tf[word])
        
        for word in tf.keys():
            self[word] = tf[word] * idf[word]

    def save(self):
        with open('tfidf.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        with open('tfidf.pkl', 'rb') as f:
            return pickle.load(f)




def clean_html(file_addr, save_addr= None):
    with open( file_addr) as reader:
        text = ' '.join(reader.readlines())
        text = text.decode('utf-8')
        result = re.findall(paragraphRegex, text)
        if result:
            text = ' '.join(result)
        text = html_regex.sub(' ', text)
        text = non_persian_regex.sub(' ', text)
        #text = puncRegex.sub(' ', text)
        text = space_regex.sub(' ', text)
        text = dot2enter_regex.sub(' \n', text)
    if save_addr is None:
        save_addr = file_addr
    with uopen(save_addr, 'w', 'utf-8') as writer :
        writer.write(text)

def clean_directory_of_htmls(dir_addr, save_dir, prefix= ''):
    #Takes dir_addr address of directory filled with html files and clean each file using utils.clean_html
    from os import listdir
    from os.path import isfile, join
    
    files = [f for f in listdir(dir_addr) if isfile(join(dir_addr, f))]
    for f in files:
        clean_html(join(dir_addr, f), join(save_dir, prefix + f))

        
def w2v_format_to_numpy_array(in_file_addr, out_file_addr):
    import numpy as np

    with open(in_file_addr) as f:
        vocab_size, embeddings_size = map(int, f.readline().split())
        inp = f.readlines()
        
    out = np.zeros((vocab_size, embeddings_size))
    for i in range(vocab_size):
        nums_str = inp[i].split()[1:]
        out[i] = map(float, nums_str)

    np.save(out_file_addr, out)


def l2(vec):
    return np.sqrt(np.sum(vec ** 2))

def cosine_dist(x, y):
    return x.dot(y)/(x.dot(x) * y.dot(y))

def read_excel_file(path):
    import xlrd
    out = list()
    book = xlrd.open_workbook(path)
    first_sheet = book.sheet_by_index(0)
    for i in range(first_sheet.nrows):
        out.append( first_sheet.row_values(i))
    return out

def keep_english(file_addr):
    with open(file_addr) as f:
        lines = [l for l in f.readlines() if l != '']
    lines = map(lambda x: re.sub(r'[^a-zA-Z ]', '', x), lines)  # removes non-english characters
    with open(file_addr, 'w') as f:
        f.writelines(lines)
def keep_english_dir(dir_name):
    for doc in os.listdir(dir_name):
        keep_english(os.path.join(dir_name, doc))


