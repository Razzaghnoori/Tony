import core
import os
import initialize

while True:
    if not os.path.exists("Phrases") and not os.path.exists("tfidf.pkl"):
        print "intilizing"
        initialize.start()

    print "how can i help you ?"
    question = raw_input()

    tony = core.Tony('word2vec/400', 'tfidf', 'Phrases', tfidf_factor=0.5)

    answer = tony.answer(question)[0]
    if not answer:
        print answer
    else:
        print "i'm not sure , i will ask him !"
