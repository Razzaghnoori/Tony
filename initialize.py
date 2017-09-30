from utilities import tools, nlp
import use
from os import mkdir
from os.path import exists, isdir
from gensim.models import Word2Vec


def start():
    tools.keep_english_dir('Files/')
    use.train_bigram_transformer(addr='Files/')
    if not isdir('word2vec'):
        mkdir('word2vec')
    if exists('word2vec/400'):
        model = Word2Vec.load('word2vec/400')
    else:
        model = use.train_word2vec('Files/', vector_size=400, use_bigram_transform=True)
        model.save('word2vec/400')
    tfidf = nlp.tfidf()
    tfidf.generate(model, 'Files/')
    tfidf.save()
