#use pre-trained word embeddings - word2vec
import gensim
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('D:\datasets\word2vec\GoogleNews-vectors-negative300.bin', binary=True)
#get the vector of any word in Google News Corpus
vector = model['London']
print(vector)
#get words similar to a given word
print("similar: ", model.most_similar('London'))



