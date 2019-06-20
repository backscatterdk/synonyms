"""
This script assumes the text column is called "text".
"""

import pandas as pd
import stanfordnlp
import os
import string
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords

nltk.download('stopwords')

"""
Choose which types of words (eg. nouns, verbs) are desired.
For POS tags, see https://universaldependencies.org/u/pos/
"""
wanted_pos = ['NOUN']


def clean_text(text):
    """
    Remove punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def get_lemma(token):
    """
    Get the lemma from the tokeniszed sentences
    """
    return [word.lemma for sent in token.sentences
            for word in sent.words]


def remove_stop(words):
    """
    Remove stop words
    """
    return [word for word in words if word not in stop_words]


def filter_pos(token):
    """
    This is for filtering based on word type
    """
    filtered = []
    for sent in token.sentences:
        filtered.extend([word.lemma for word in sent.words
                         if word.upos in wanted_pos])
    filtered = list(set(filtered))
    return filtered

def remove_punc(words):
    """
    Removes punctuation and lowercases.
    """
    out = []
    for w in words:
        out.append(''.join(e.lower() for e in w if e.isalnum()))
    return out

# Download dansk model for nlp.
if not os.path.exists(os.path.join(os.environ['HOME'],
                      'stanfordnlp_resources', 'da_ddt_models')):
    stanfordnlp.download('da')


# Set up nlp pipeline
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,lemma,pos', lang='da')

# Read data. Change to correspond.
df = pd.read_csv('parents_full.csv')

# For progress bar
tqdm.pandas()

# Get get stop words
stop_words = stopwords.words('danish')

df['tokens'] = df['text'].progress_apply(lambda text: nlp(text))
df['lemmas'] = df['tokens'].apply(get_lemma)
df['lemmas_string'] = df['lemmas'].apply(lambda x: " ".join(x))
df['without_stop'] = df['lemmas'].apply(remove_stop)
df['filtered'] = df['tokens'].apply(filter_pos)
df['filtered'] = df['filtered'].apply(remove_punc)
df['filtered'] = df['filtered'].apply(lambda x: ", ".join(x))
df.drop(['tokens', 'lemmas', 'lemmas_string', 'without_stop'],
        axis=1, inplace=True)

df.to_csv('nouns_output.csv')
