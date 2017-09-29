import core
import datetime


while True:
    question = raw_input()
    #before = datetime.datetime.now()
    tony = core.Tony('word2vec/400', 'tfidf', 'Phrases', tfidf_factor=0.5)
    #print 'How can I help you?'
    print tony.answer(question)[0]
    #after = datetime.datetime.now()
    #print after - before
