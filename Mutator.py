# coding: utf-8

import numpy as np
import requests
from stanfordcorenlp import StanfordCoreNLP
from config import *
import nltk.stem as ns

# config api url
base_url = api_url


class ActiveMutator:
    def __init__(self, sensitive_attribute):
        self.sensitive_attribute = sensitive_attribute
        self.identify_group = self.create_identify_group()

    def create_identify_group(self):
        if self.sensitive_attribute == 'gender':
            return {"female", "male"}
        if self.sensitive_attribute == 'lgbt':
            return {"gay", "lesbian", "bisexual"}

    def create_active_candidates(self, token: str):
        return {f'{adj} {token}' for adj in self.identify_group}


class AnalogyMutator:
    antonym_cache = dict()
    formOf_cache = dict()
    analogy_mutation = dict()

    def __init__(self, sensitive_attribute, enable_pre_fetch=False, model=None):
        self.sensitive_attribute = sensitive_attribute
        if sensitive_attribute == 'gender':
            
            self.model = model
            self.dist = lambda x, y: np.sum((x-y)**2)
            self.pre_fetch = {}
            if enable_pre_fetch:
                with open("word_pairs.txt") as wp:
                    for l in wp.readlines():
                        l = l.strip().split()
                        self.pre_fetch[l[0]] = l[1]
                        self.pre_fetch[l[1]] = l[0]

    def create_analogy_candidates(self, token: str, strict_mode = False):
        
        if self.sensitive_attribute == 'gender':
            if token not in self.model:
                return set()
            if token in self.pre_fetch:
                return {self.pre_fetch[token]}
            # 1 denotes male, 0 denotes female
            gender = 1 if self.dist(self.model[token], self.model["male"]) < \
                          self.dist(self.model[token], self.model["female"]) else 0
            # in this implementation we empirically enforce i[1]>0.7
            if gender == 1:
                candidates1 = {i[0].lower() for i in
                               self.model.most_similar(positive=["woman", token], negative=["man"])
                               if i[1]>.7}
            else:
                candidates1 = {i[0].lower() for i in
                               self.model.most_similar(positive=["man", token], negative=["woman"])
                               if i[1]>.7}
            
            # strict mode, enhance candidate by incroporating KG
            candidates2 = AnalogyMutator.antonym(token) | AnalogyMutator.formof(token) if strict_mode else candidates1

            if len(candidates1 & candidates2) != 0:
                AnalogyMutator.analogy_mutation[token] = candidates1 & candidates2
                return candidates1 & candidates2
            else:
                return set()
        return set()

    @staticmethod
    def antonym(token: str):
        start = '/c/en/' + token.lower()
        antonym_cache = AnalogyMutator.antonym_cache
        if start in antonym_cache:
            return antonym_cache[start]
        for i in range(3):
            try:
                response = requests.get(f'{base_url}/query?start={start}&rel=/r/Antonym')
                obj = response.json()
                edges = obj['edges']
                if edges:
                    end_nodes = [edge['end']['@id'] for edge in edges if edge['weight'] >= 1]
                    antonym_cache[start] = set([n.split("/")[3] for n in end_nodes])
                    return antonym_cache[start]
                else:
                    return set()
            except Exception as e:
                print(f'Query on {start} of Antonym edge failed {["once", "twice", "three times"][i]}.', e)
        print(f'Query on {start} of Antonym edge failed.')
        return set()

    @staticmethod
    def formof(token: str):
        start = '/c/en/' + token.lower()
        formOf_cache = AnalogyMutator.formOf_cache
        if start in formOf_cache:
            return formOf_cache[start]
        for i in range(3):
            try:
                response = requests.get(f'{base_url}/query?start={start}&rel=/r/FormOf')
                obj = response.json()
                edges = obj['edges']
                if edges:
                    end_nodes = [edge['end']['@id'] for edge in edges if edge['weight'] >= 1]
                    formOf_cache[start] = set([n.split("/")[3] for n in end_nodes])
                    return formOf_cache[start]
                else:
                    return set()
            except Exception as e:
                print(f'Query on {start} of FormOf edge failed {["once", "twice", "three times"][i]}.', e)
        print(f'Query on {start} of FormOf edge failed.')
        return set()


isA_cache = dict()

def isa(start="", limit=10):
    if start in isA_cache:
        return isA_cache[start]
    for i in range(3):
        try:
            response = requests.get(f'{base_url}/query?start={start}&rel=/r/IsA&limit={limit}')
            obj = response.json()
            edges = obj['edges']
            end_nodes = [edge['end']['@id'] for edge in edges if edge['weight'] >= 1]
            isA_cache[start] = ["/".join(n.split("/")[:4]) for n in end_nodes]
            return isA_cache[start]
        except Exception as e:
            print()
            print(f'Query on {start} of IsA edge failed {["once", "twice", "three times"][i]}.', e)
    print(f'Query on {start} of IsA edge failed.')
    return []


# check if the token is human
# load pre-defined rules
def check_human(token, debug=False):
    if token in insensitive_blacklist or '/c/en/' + token in no_human_indicator:
        return False
    token = lemmatizer.lemmatize(token.lower(), 'n')
    tokens = ['/c/en/' + token]
    visited = set()
    # check if the token is a type of person/people/human with search depth of 3
    topK2 = [8, 4]
    for i in range(2):
        next_round_token = []
        for w in tokens:
            if w not in visited:
                visited.add(w)
                end_nodes = isa(start=w, limit=30)
                if debug: print(w, end_nodes)
                if len(no_human_indicator & set(end_nodes)) > 0:
                    return False
                if len(human_indicator & set(end_nodes)) > 0:
                    # print(token, w)
                    return True
                if len(neutral_indicator & set(end_nodes)) > 0:
                    return False
                next_round_token += end_nodes[:topK2[i]]
        tokens = next_round_token
    return False

def make_mutation(pos, replacement, index):
    s = ""
    for i in range(len(pos)):
        if i == index:
            s += replacement + " "
        else:
            s += pos[i][0] + " "
    return s

# word tagging and singularize
lemmatizer = ns.WordNetLemmatizer()
print("stanford corenlp model loading")
nlp = StanfordCoreNLP('dependency/stanford-corenlp-full-2018-10-05')
print("stanford corenlp model loaded")
part_of_speech = lambda x: nlp.pos_tag(x.strip())


def create_sentence_candidates(sentence: str, ana, act):
    ana_candidates = set()
    act_candidates = set()

    pos = part_of_speech(sentence)
    human_token = []
    for temp, index in zip(pos, range(len(pos))):
        token, tag = temp
        # print(token, tag)
        if len(token) == 1 or tag not in {'NN', 'NNS'}:
            continue
        if check_human(token):
            human_token.append((token, tag, index))

    for token, tag, index in human_token:
        if act is not None and index == 0 or pos[index-1][1] not in {'NN', 'JJ', 'JJR', 'JJS'}:
            act_candidates |= {make_mutation(pos, replacement, index)
                                for replacement in act.create_active_candidates(token)
                                if replacement is not None}

        if tag == 'NNS':
            word = lemmatizer.lemmatize(token.lower(), 'n')
        else:
            word = token.lower()

        if act is not None:
            ana_candidates |= {make_mutation(pos, replacement, index)
                                for replacement in ana.create_analogy_candidates(word)
                                if replacement is not None}

    return ana_candidates, act_candidates

if __name__ == "__main__":
    import gensim.downloader as api
    print("word2vec model loading")
    word2vec = api.load("word2vec-google-news-300")
    print("word2vec model loaded")
    ana = AnalogyMutator("gender", model=word2vec)
    pair_list = []
    with open("word_pairs.txt") as wp:
        for l in wp.readlines():
            l = l.strip().split()
            pair_list.append((l[0],l[1]))
    for p in pair_list:
        print(p[0], ana.create_analogy_candidates(p[0]))
        print(p[1], ana.create_analogy_candidates(p[1]))
    while True:
        s = input()
        print(ana.create_analogy_candidates(s))
    