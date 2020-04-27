# MT-NLP

Implementation of paper ["Metamorphic Testing and Certified Mitigation of Fairness Violations in NLP Models"](https://placeholder). Pingchuan Ma, Shuai Wang, and Jin Liu. In Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI '20), 2020.

Please cite as:

```bibtex
@incollection{mtnlp,
title = {Metamorphic Testing and Certified Mitigation of Fairness Violations in NLP Models},
author = {Pingchuan Ma and Shuai Wang and Jin Liu},
booktitle = {29th International Joint Conference on Artificial Intelligence},
year = {2020},
}
```

# Dependency

Note that all codes are written in Python 3.6. Please download the [file](https://drive.google.com/file/d/1nNzFkDw2CQGq9EFIjnuWMDdisb0IYGFn/view?usp=sharing) and extract in the `./dependency` folder.

## Python Library

- tensorflow (1.8.0)
- keras (2.2.4)
- sklearn (0.20.3)
- numpy (1.16.2)
- gensim (3.8.1) (Please download `word2vec-google-news-300` model after installation)
- stanfordcorenlp (3.9.1.1)
- nltk (3.5) (Please download `WordNet` after installation)

Since Stanford Corenlp is one of our dependencies, `jdk` is required.
In addition, to evaluate the fluency score of mutations, please follow the instruction of [pytorch-human-performance-gec](https://github.com/rgcottrell/pytorch-human-performance-gec) to install dependencies.

# Demo

## Usage
```
python demo.py -h
```

Output:
```
usage: demo.py [-h] [-s S] [-k K] [-e E]

Mutate sentence and mitigate violations

optional arguments:
  -h, --help  show this help message and exit
  -s S        file of seed sentence
  -k K        mitigation parameter k
  -e E        mitigation parameter epsilon
```

## Example

```
python demo.py -s example.txt -k 15 -e 2.0
```

Output (it will download necessary word embedding models in the first time):
```
# testcases (k):  15
original score:  [0.66590524]
Text in 'example.txt'
# violations:  5
# violations (after mitigation):  3
```

# Customization

Note that all the customization are disabeld by default, which is consistent with our paper.

## Mutation Enhancement

For the reason that the correctness of word analogy in the word embedding model is not guaranteed, we employ a Knowledge Graph-based post-process mechanism to validate generated tokens and rule out potentially error ones (cf. `line 68` in `Mutator.py`).

## Pre-fetch

In case invoking remote KG could be slow sometimes, we design a local analogy pair list w.r.t. "gender" to pre-fetch analogy mutations without invoking word embedding model and knowledge graph api. Note that all pairs in the local list can be mapped by KG as well. We also note that there are some corner cases, e.g., `wizard`-`witch` and `conductor`-`conductress`, that cannot be mapped by our algorithm. In that sense, users can manually add analogy pairs in the local file to enrich mapping rules.

# Dataset

The dataset is derived from [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

Download Link: [Google Drive](https://drive.google.com/file/d/1yya4l3Um6bF84gqCyeRLjsOt4Qw-6TNl/view?usp=sharing)