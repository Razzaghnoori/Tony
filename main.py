import core
import os
import initialize

while True:
    if not (os.path.exists('word2vec/400') and os.path.exists('Phrases') and os.path.exists('tfidf.pkl')):
        print "Intilizing"
        initialize.start()

    print "How can I help you?"
    question = raw_input()

    tony = core.Tony('word2vec/400', 'tfidf.pkl', 'Phrases', tfidf_factor=0.5)

    answer = tony.answer(question)
    if len(answer) > 0:
        print answer
    else:
        print "I am not sure. I will inform Mr. Razzaghnoori."

