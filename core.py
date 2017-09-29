import gensim
import utilities
import numpy as np
import datetime
from os import listdir
from os.path import join, isfile

class Tony():
    def __init__(self, model_addr, tfidf_addr, phrases_addr, knowledge_dir='knowledge', threshold=0.01, tfidf_factor=1):
        self.knowledge_dir = knowledge_dir
        self.threshold = threshold
        self.tfidf_factor = tfidf_factor
        self.knowledge = dict()

        start = datetime.datetime.now()
        self.model = gensim.models.Word2Vec.load(model_addr)
        print "time to load model: %s"%(datetime.datetime.now() - start)
        start = datetime.datetime.now()
        self.tfidf = utilities.tfidf()
        self.tfidf.load()
        print "time to load tfidf: %s"%(datetime.datetime.now() - start)
        start = datetime.datetime.now()
        self.bigram_transformer = gensim.models.phrases.Phraser.load(phrases_addr)
        print "time to load bigram transformer: %s"%(datetime.datetime.now() - start)
        start = datetime.datetime.now()
        self.code_knowledge()
        print "time to load knowledge: %s"%(datetime.datetime.now() - start)

    def set_knowledge_dir(self, knowledge_dir):
        self.knowledge_dir = knowledge_dir

    def set_tfidf_factor(self, tfidf_factor):
        self.tfidf_factor = tfidf_factor

    def new_representation(self, questions):
        m = len(questions)
        n = self.model.vector_size
        X = np.zeros((m,n))
        for i, question in enumerate(questions):
            X[i] = np.sum([self.model[word]/utilities.l2(self.model[word]) * (self.tfidf[word] if self.tfidf.get(word) else 1) ** self.tfidf_factor
                           for word in self.bigram_transformer[question.split()] if word in self.model.wv.vocab], axis= 0)
        return X

    def code_knowledge(self):
        files = [f for f in listdir(self.knowledge_dir) if isfile(join(self.knowledge_dir, f))]
        for file_ in files:
            with open(join(self.knowledge_dir, file_)) as f:
                self.knowledge[file_] = self.new_representation(f.readlines())


    def answer(self, question):
        answers = list()
        summary = list()
        question_vec = self.new_representation([question]).reshape(-1)

        for doc_name, doc_matrix in self.knowledge.iteritems():
            for sentence_num, sentence in enumerate(doc_matrix):
                sim = utilities.cosine_dist(question_vec, sentence.reshape(-1))
                if sim > self.threshold:
                    answers.append((doc_name, sentence_num, sim))

        sorted_answers = sorted(answers, key= lambda x: x[2], reverse=True)

        for (doc_name, sentence_num, _) in sorted_answers:
            with open(join(self.knowledge_dir, doc_name)) as f:
                summary.append(f.readlines()[sentence_num])

        return summary
