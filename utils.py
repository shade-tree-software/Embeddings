# -*- coding: utf-8 -*-
import re
import string

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from textSummarizer import summarize_text

def filter_text(text):
    done = False
    while not done:
        init_len = len(text)
        text = re.sub(r'\s+\.', '.', text) # remove any whitespace before a period
        text = re.sub(r'\u200c', '', text) # remove zero-width-non-joiner
        text = re.sub(r'\s{2,}', ' ', text) # convert two or more consecutive whitespace chars to a single space
        text = re.sub(r'\.{2,}', '.', text) # convert two or more consecutive periods to a single period
        text = re.sub(r'[\-\'\.=:#$]{2,}', '', text) # remove two or more consecutive punctuation chars
        text = re.sub(r'\s[\-\'\.=:#$]\s', ' ', text) # remove lone punctuation surrounded by whitespace
        text = re.sub(r'\d+\.*\s\d+', '', text) # remove two numbers separated by spaces
        done = len(text) == init_len
    return text

def is_jumble(word):
    # do not allow words that look like jumbles of digits and non-digits
    return re.search(r'\d', word) and re.search(r'\D', word) and not re.search(r'[\.\-g-zG-Z:]', word)

def is_hex_consonants(word):
    # do not allow words that contain only hex consonants
    return len(word) > 2 and not re.search(r'[^bcdfBCDF]', word)

def is_hex_value(word):
    # do not allow words that start with 'x' followed by four hex chars
    return re.search(r'^x[a-zA-Z0-9]{4}', word)

def is_bad_word(word):
    return len(word) > 15 or is_jumble(word) or is_hex_consonants(word) or is_hex_value(word)

def get_words(text):
    text = re.sub(r'[,!?;]', '.', text) # convert common punctuation to periods
    text = re.sub(r'=[A-F0-9]{2}', ' ', text) # remove inline hex chars
    text = re.sub(r'["%&()*+/<>[\\\]^_`{|}~=]', '', text) # remove certain punctuation
    text = filter_text(text)
    words = [word for word in word_tokenize(text) if not is_bad_word(word)]
    return words

def simplify_text(text, summarize = False):
    words = get_words(text)
    text = ' '.join(words)
    text = filter_text(text)
    if summarize:
        text = summarize_text(text)
    return text

def process_text(text):
    words = get_words(text)
    # stem words and remove stopwords
    stem = PorterStemmer().stem
    en_stopwords = stopwords.words('english')
    return [stem(word.lower()) for word in words if word not in en_stopwords]

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10    
    dot = np.dot(A, B)
    normb = np.linalg.norm(B)
    
    if len(A.shape) == 1: # If A is just a vector, we get the norm
        norma = np.linalg.norm(A)
        cos = dot / (norma * normb)
    else: # If A is a matrix, then compute the norms of the word vectors of the matrix (norm of each row)
        norma = np.linalg.norm(A, axis=1)
        epsilon = 1.0e-9 # to avoid division by 0
        cos = dot / (norma * normb + epsilon)
        
    return cos
