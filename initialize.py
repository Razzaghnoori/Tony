import utilities
import use


utilities.keep_english_dir('Files/')
use.train_bigram_transformer(addr='Files/')
use.train_word2vec('Files/', vector_size=400, use_bigram_transform=True)
tfidf = utilities.tfidf()
tfidf.generate('Files/')
tfidf.save('tfidf')

