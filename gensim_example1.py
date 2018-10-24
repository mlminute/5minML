#create your own word2vec model
import gensim
from gensim.models import Word2Vec

sentences = [['London', 'is', 'the', 'capital', 'of', 'England'], ['Paris', 'is', 'the', 'capital', 'of', 'France']]
# train word2vec on the two sentences
model = Word2Vec(sentences, min_count=1)	
print(model['capital'])
print(model['capital'].shape)