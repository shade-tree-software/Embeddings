import pdb
import pickle
import string
import sys
from os import listdir
from os.path import isfile, join

import nltk
import numpy as np
from nltk.corpus import stopwords

from utils import (cosine_similarity, process_text)

# en_embeddings = pickle.load(open("data/en_embeddings.p", "rb"))
EMBEDDINGS_FILE = "data/glove.6B.300d.p"
en_embeddings = pickle.load(open(EMBEDDINGS_FILE, "rb"))

doc_dir = sys.argv[1]
query = sys.argv[2]

def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - the indices of the top k closest vectors
    """
    cos_similarities = []
    # get cosine similarity of input vector v and each candidate vector
    for candidate in candidates:
        cos_similarities.append(cosine_similarity(v, candidate))
    # sort the similarity list and get the k most similar indices    
    return np.flip(np.argsort(cos_similarities))[:k]

def get_document_embedding(text, embeddings, process_text=process_text):
    '''
    Input:
        - text: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)
    # process the document into a list of words (process the tweet)
    processed_doc = process_text(text)
    words_with_embeddings = set()
    for word in processed_doc:
        if word not in ['transfer', 'type', 'html', 'utf', 'content', 'text', 'div', 'http', 'www', 'org']:
            # add the word embedding to the running total for the document embedding
            word_embedding = embeddings.get(word, 0)
            if isinstance(word_embedding, np.ndarray):
                words_with_embeddings.add(word)
            doc_embedding += word_embedding
    return doc_embedding

def get_document_vecs(all_docs, en_embeddings, get_document_embedding=get_document_embedding):
    '''
    Input:
        - all_docs: list of strings - all documents in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of document embeddings.
        - ind2Doc_dict: dictionary with indices of docs in vecs as keys and their embeddings as the values.
    '''

    # the dictionary's key is an index (integer) that identifies a specific document
    # the value is the document embedding for that document
    ind2Doc_dict = {}

    # this is list that will store the document vectors
    document_vec_l = []

    for i, doc in enumerate(all_docs):
        doc_embedding = get_document_embedding(doc, en_embeddings)
        ind2Doc_dict[i] = doc_embedding
        document_vec_l.append(doc_embedding)

    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict

doc_files = [join(doc_dir, f) for f in listdir(doc_dir) if isfile(join(doc_dir, f))]
print(f"loading {len(doc_files)} documents")
docs = []
for doc_file in doc_files:
    with open(doc_file, "r") as f:
        docs.append(f.read())
doc_vecs, docs_by_index = get_document_vecs(docs, en_embeddings)
query_embedding = get_document_embedding(query, en_embeddings)
idx = np.argmax(cosine_similarity(doc_vecs, query_embedding))
print(docs[idx])
