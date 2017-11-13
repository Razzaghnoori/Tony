import gensim
from utilities import tools
from utilities import nlp
import numpy as np
from os import listdir
from os.path import join, isfile
import re

class Tony():
    def __init__(self, model_addr, tfidf_addr, phrases_addr, knowledge_dir='knowledge',
                 threshold=0.01, tfidf_factor=1, use_bigram=False):

        self.knowledge_dir = knowledge_dir
        self.threshold = threshold
        self.tfidf_factor = tfidf_factor
        self.use_bigram = use_bigram
        self.knowledge = dict()

        self.model = gensim.models.Word2Vec.load(model_addr)

        self.tfidf = nlp.tfidf()
        self.tfidf = self.tfidf.load()

        self.bigram_transformer = gensim.models.phrases.Phraser.load(phrases_addr)
        self.char_regex = re.compile("\w+")
        #self.code_knowledge()

    def set_knowledge_dir(self, knowledge_dir):
        self.knowledge_dir = knowledge_dir

    def set_tfidf_factor(self, tfidf_factor):
        self.tfidf_factor = tfidf_factor

    def new_representation(self, questions):
        m = len(questions)
        n = self.model.vector_size
        X = np.zeros((m, n))
        for i, question in enumerate(questions):
            question = question.lower()
            question = ' '.join(self.char_regex.findall(question))
            if self.use_bigram:
                words = self.bigram_transformer[question.split()]
            else:
                words = question.split()

            X[i] = np.sum([self.model[word] / tools.l2(self.model[word]) * self.tfidf[word] ** self.tfidf_factor
                           for word in words if word in self.model.wv.vocab], axis=0) / len(words)
        return X

    def code_knowledge(self):
        files = [f for f in listdir(self.knowledge_dir) if isfile(join(self.knowledge_dir, f))]
        for file_ in files:
            with open(join(self.knowledge_dir, file_)) as f:
                lines = f.readlines()
                lines = map(str.lower, lines)
                lines = [' '.join(self.char_regex.findall(line)) for line in lines]
                self.knowledge[file_] = self.new_representation(lines)

    def answer(self, question):
        question = question.lower()
        question_word_set = set([word for word in question.split() if word not in self.model])
        best_similarity = -np.inf
        question_matrix = np.vstack([self.model[word]
                                     for word in question.split()
                                     if self.model.wv.vocab.get(word)])

        question_matrix /= np.linalg.norm(question_matrix, axis=1).reshape(-1, 1)

        for file_ in listdir(self.knowledge_dir):
            with open(join(self.knowledge_dir, file_)) as handle:
                for sentence in handle.readlines():
                    sentence = sentence.lower()
                    sentence_list_of_vectors = [self.model[word]
                                                for word in sentence.split()
                                                if self.model.wv.vocab.get(word)]
                    if len(sentence_list_of_vectors) == 0:
                        continue
                    sentence_matrix = np.vstack(sentence_list_of_vectors)
                    sentence_matrix /= np.linalg.norm(sentence_matrix, axis=1).reshape(-1, 1)
                    sentence_matrix = sentence_matrix.T
                    
                    similarity_matix = np.dot(question_matrix, sentence_matrix)
                    similarity = np.mean(np.max(similarity_matix, axis=1)) + \
                                 len(question_word_set.intersection([word for word in set(sentence.split()) if word not in self.model]))

                    if similarity > best_similarity:
                        best_similarity = similarity
                        result = sentence
                        
        print best_similarity
        return result

                    
