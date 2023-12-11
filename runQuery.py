import pdb
import pickle
import string
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import json

from vertexai.preview.language_models import TextEmbeddingModel

from utils import (cosine_similarity, process_text)

# Match query text with the document that has the closest embedding.
#
# Documents are read from a JSONL file.  If the file also contains embeddings
# for each document, then they are assumed to be Google embeddings and the
# query text embedding is requested from Google.  If the file contains no
# embeddings then all embeddings are generated locally.
#
# Local embeddings are created by summing embeddings for each word, in which
# case any context based on word order is lost.  Thus, local embeddings may
# not be as accurate as embeddings generated by Google.
#
# If a JSONL file is not specified, documents and embeddings are loaded from
# pickle files.

LOCAL_DOCS_PICKLE = "data/docs_local.p"
GOOGLE_DOCS_PICKLE = "data/docs_google.p"
LOCAL_DOC_VECS_PICKLE = "data/docVecs_local.p"
GOOGLE_DOC_VECS_PICKLE = "data/docVecs_google.p"
LOCAL_WORD_VECS_PICKLE = "data/glove.6B.300d.p"
# LOCAL_WORD_VECS_PICKLE = "data/local_word_vecs.p"

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

def get_document_embedding_verbose(text, embeddings, process_text=process_text):
    '''
    Input:
        - text: a string
        - embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)
    # process the document into a list of words (process the tweet)
    processed_doc = process_text(text)
    words_with_embeddings = set()
    for word in processed_doc:
        # add the word embedding to the running total for the document embedding
        word_embedding = embeddings.get(word, 0)
        if isinstance(word_embedding, np.ndarray):
            words_with_embeddings.add(word)
        doc_embedding += word_embedding
    return doc_embedding, words_with_embeddings, processed_doc

def get_document_embedding(text, embeddings, process_text=process_text):
    return get_document_embedding_verbose(text, embeddings, process_text)[0]

def get_document_vecs(all_docs, embeddings, get_document_embedding=get_document_embedding):
    '''
    Input:
        - all_docs: list of strings - all documents in our dataset.
        - embeddings: dictionary with words as the keys and their embeddings as the values.
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
        doc_embedding = get_document_embedding(doc, embeddings)
        ind2Doc_dict[i] = doc_embedding
        document_vec_l.append(doc_embedding)

    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict

# Load data based on command line options
local_word_vecs = None
query = sys.argv[1]
try:
    docs_jsonl = sys.argv[sys.argv.index("-j") + 1]
except:
    docs_jsonl = None
if docs_jsonl:
    with open(docs_jsonl, "r") as f:
        docs_info = [json.loads(doc_info) for doc_info in f.readlines()]
    if "instance" in docs_info[0] and "predictions" in docs_info[0]:
        docs = []
        doc_vecs_l = []
        for doc_info in docs_info:
            docs.append(doc_info["instance"]["content"])
            doc_vecs_l.append(doc_info["predictions"][0]["embeddings"]["values"])
        doc_vecs = np.vstack(doc_vecs_l)
        print(f"Loaded {len(docs)} docs and embeddings from {docs_jsonl}")
    elif "content" in docs_info[0]:
        docs = [doc_info["content"] for doc_info in docs_info]
        print(f"Loaded {len(docs)} docs from {docs_jsonl}.  No embeddings found.  Generating all embeddings locally.")
        local_word_vecs = pickle.load(open(LOCAL_WORD_VECS_PICKLE, "rb"))
        doc_vecs, _ = get_document_vecs(docs, local_word_vecs)
    else:
        print("could not read input file")
        exit(0)
else:
    if "-g" in sys.argv:
        try:
            docs = pickle.load(open(GOOGLE_DOCS_PICKLE, "rb"))
        except:
            print("No JSONL file specified and cannot find docs pickle file")
            exit(0)
        try:
            doc_vecs = pickle.load(open(GOOGLE_DOC_VECS_PICKLE, "rb"))
        except:
            print("Google option specified but cannot find Google doc embeddings pickle file")
            exit(0)
    else:
        try:
            docs = pickle.load(open(LOCAL_DOCS_PICKLE, "rb"))
        except:
            print("No JSONL file specified and cannot find docs pickle file")
            exit(0)
        try:
            doc_vecs = pickle.load(open(LOCAL_DOC_VECS_PICKLE, "rb"))
        except:
            print("No JSONL file specified and cannot find local doc embeddings pickle file")
            exit(0)
        try:
            local_word_vecs = pickle.load(open(LOCAL_WORD_VECS_PICKLE, "rb"))
        except:
            print("Cannot find local word embeddings pickle file")
            exit(0)

# generate embeddings for query text
if local_word_vecs:
    query_embedding, words_with_embeddings, _ = get_document_embedding_verbose(query, local_word_vecs)
    print(f"Query words: {words_with_embeddings}")
else:
    print("Requesting query text embedding from Google")
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
    query_embedding = model.get_embeddings([query])[0].values

# print best match
idx = np.argmax(cosine_similarity(doc_vecs, query_embedding))
print(f"Best document match for query text:\n{docs[idx]}")
if local_word_vecs:
    print("Words from document that have embeddings:")
    print(get_document_embedding_verbose(docs[idx], local_word_vecs)[1])
else:
    print("Cleaned words from document:")
    print(set(process_text(docs[idx])))

if docs_jsonl:
    print(f"Saving docs as pickle file")
    if local_word_vecs:
        pickle.dump(docs, open(LOCAL_DOCS_PICKLE, "wb"))
        print(f"Saving local doc embeddings as pickle file")
        pickle.dump(doc_vecs, open(LOCAL_DOC_VECS_PICKLE, "wb"))
    else:
        pickle.dump(docs, open(GOOGLE_DOCS_PICKLE, "wb"))
        print(f"Saving Google doc embeddings as pickle file")
        pickle.dump(doc_vecs, open(GOOGLE_DOC_VECS_PICKLE, "wb"))

