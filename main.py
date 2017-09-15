import core
import datetime

tony = core.Tony('word2vec/400', 'tfidf', 'Phrases', tfidf_factor=0.5)

print 'How can I help you?'
while True:
    before = datetime.datetime.now()
    print tony.answer(raw_input())[0]
    after = datetime.datetime.now()
    print after - before
