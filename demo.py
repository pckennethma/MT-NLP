#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser(description='Mutate sentence and mitigate violations')

parser.add_argument('-s', help='file of seed sentence')
parser.add_argument('-k', type=int, help='mitigation parameter k')
parser.add_argument('-e', type=float, help='mitigation parameter epsilon')

args = parser.parse_args()

import pandas as pd
import pickle, os
import gensim.downloader as api
from Mutator import AnalogyMutator, ActiveMutator, create_sentence_candidates
from fluency_scorer import FluencyScorer
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np



# load word2vec embedding
print("word2vec model loading")
word2vec = api.load("word2vec-google-news-300")
print("word2vec model loaded")

# load scorer
p, d = "dependency/wiki103/wiki103.pt", "dependency/wiki103"
fluency_scorer = FluencyScorer(p, d)

# load cnn model
VOCAB_SIZE = 30000
MAX_LEN = 500

model = load_model('model/cnn_model.h5')
with open('model/cnn_model_tokenizer.pickle', 'rb') as f:
    imdb_tokenizer = pickle.load(f)

ana = AnalogyMutator("gender", model=word2vec)
act = ActiveMutator("gender")

def mutate(sentence, epsilon):
    ana_candidates, act_candidates = create_sentence_candidates(sentence, ana, act)
    candidates = list(ana_candidates) + list(act_candidates)
    
    if len(candidates) > args.k:
        candidates_with_fluency = [(fluency_scorer.score_sentence(c).item(), c) for c in candidates]
        candidates_with_fluency.sort(key=lambda x:x[0], reverse=True)
        candidates = [c[1] for c in candidates_with_fluency[:args.k]]
    
    sentences_seq = imdb_tokenizer.texts_to_sequences([sentence] + candidates)
    sentences_vec = sequence.pad_sequences(sentences_seq, maxlen=MAX_LEN, padding="post", value=0)
    predictions = model.predict(sentences_vec) - .5
    original_score, testcase_scores = predictions[0], predictions[1:]

    print("# testcases (k): ", len(candidates))
    print("original score: ", original_score + .5)
    print(sentence)
    num_violations = sum([1 for i in range(len(testcase_scores)) if testcase_scores[i] * original_score < 0])
    print("# violations: ", num_violations)
    print("# violations (after mitigation): ", mitigation(predictions, epsilon))
    
def mitigation(predictions, epsilon = 2):
    num_violations = 0
    ground_truth = predictions[0]
    k = len(predictions)
    for i in predictions[1:]:
        score = np.sum(predictions) / (np.e**epsilon + k) + np.e**epsilon * i / (np.e**epsilon + k)
        if score * ground_truth < 0:
            num_violations += 1
    return num_violations

with open(args.s) as f:
    s = f.readline()

mutate(s, epsilon=args.e)