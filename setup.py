import pickle
import numpy as np
import nltk

nltk.download('stopwords')

# cd data
# wget https://downloads.cs.stanford.edu/nlp/data/glove.840B.300D.zip
# unzip -q glove.840B.300D.zip
 
# create embeddings pickle file
en_embeddings = {}
with open("data/glove.840B.300d.txt") as f:
    for line in f:
        word, coefs = line.split(" ", maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        en_embeddings[word] = coefs
pickle.dump(en_embeddings, open("data/glove.840B.300d.p", "wb"))

