# Brug *word2vec_tol* conda env til dette script.
# Make terms list

import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from Levenshtein import distance


def w2v_similarity(row):
    """
    Test w2v-lighed mellem to ord
    """
    try:
        return w2v.similarity(row[0], row[1])
    except:
        return np.nan


def lev_similarity(row):
    """
    Test w2v-lighed mellem to ord
    """
    try:
        return distance(row[0], row[1])
    except:
        return np.nan


# Change these values to allow more or less words
LEV_CUTOFF = 2
W2V_CUTOFF = 0.65

tqdm.pandas()

data = pd.read_csv('nouns_output.csv')

# Make flat list of terms
org_terms = []
for lst in data['filtered']: # ASSUMES COL NAME 'filtered' FROM nlp_tool.py
    if not type(lst) == float:
        for w in lst.split(', '):
            org_terms.append(w)
org_terms = list(set(org_terms))

df = pd.DataFrame({'terms': org_terms})

df.dropna(inplace=True)

# Lav en en ny df med alle ordkombinationer
terms = pd.DataFrame(list(itertools.combinations(df['terms'], 2)))

# W2V SIMILARITY

'''
TO TRAIN A NEW MODEL

import multiprocessing
import gensim, logging
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


wiki = WikiCorpus('own_tool/data/da/dawiki-latest-pages-articles.xml.bz2',
                  lemmatize=False, dictionary={})
sentences = list(wiki.get_texts())
params = {'size': 200, 'window': 10, 'min_count': 10,
          'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3,}
w2v = Word2Vec(sentences, **params)
'''

w2v = Word2Vec.load("danish_wiki.model") # Replace with above if training.

terms['w2v'] = terms.progress_apply(w2v_similarity, axis=1)

terms.sort_values(by='w2v', ascending=False).head(15)

# Lav en df med de originale ord som indeks,
# hvis de er over en vis værdi (CUTOFF) i w2v-lighed.
# Kolonne-1 er de ord, der har lighed over CUTOFF med ordene i indeks.
merge = pd.DataFrame(terms[terms['w2v'] > W2V_CUTOFF]
                     .groupby(0)[1].apply(lambda x: ", ".join(x)))

# Merge-dataframen har kun de ord, der overstiger CUTOFF.
# Derfor skal vi merge den tilbage på den oprindelige dataframe.
df = df.merge(right=merge, left_on='terms', right_index=True, how='left')

# LEVENSHTEIN DISTANCE
terms['lev'] = terms.progress_apply(lev_similarity, axis=1)
merge = pd.DataFrame(terms[terms['lev'] < LEV_CUTOFF]
                     .groupby(0)[1].apply(lambda x: ", ".join(x)))
df = df.merge(right=merge, left_on='terms', right_index=True, how='left')
df.columns = ['terms', 'w2v_words', 'lev_words']
df.to_csv('word2vec_output.csv')
