"""
This script takes in a csv with a column of text and generates
a copy of that csv with words of a given PoS-tag (eg nouns, verbs etc)
filtered in the column "filtered"

For linux systems, you may have to run this in your terminal first
to get the picking options to work

$ export TERM=linux
$ export TERMINFO=/bin/zsh

"""

import os
import string
import pandas as pd
import stanfordnlp
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from pick import pick


def main(data):
    nltk.download('stopwords')

    # Pick which PoS tags you want
    postag_title = 'Please POS tags (SPACE to mark, ENTER to continue)'
    postags = ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ',
               'X', 'NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ']
    wanted_pos = pick(postags, postag_title,
                      multi_select=True, min_selection_count=1)
    wanted_pos = [pos[0] for pos in wanted_pos]

    # Pick language
    lang_title = 'Please choose which language the text is in.'
    langs = ['en', 'da', 'other']
    lang, lang_title = pick(langs, lang_title)
    if lang == 'other':
        lang = input('Please input language code \
	(see stanfordnlp.github.io/stanfordnlp/models.html)')

    # Download model for nlp.
    if not os.path.exists(os.path.join(os.environ['HOME'],
                                       'stanfordnlp_resources', f'{lang}_ddt_models')):
        stanfordnlp.download(lang)

    # Set up nlp pipeline
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,lemma,pos', lang=lang)

    # pick column for terms
    column_title = 'Please chose which column contains the words.'
    columns = data.columns
    column, column_title = pick(columns, column_title)

    # For progress bar
    tqdm.pandas(desc="Tokenizing and POS-tagging...")

    data['tokens'] = data[column].progress_apply(lambda text: nlp(text))
    data['lemmas'] = data['tokens'].apply(get_lemma)
    data['lemmas_string'] = data['lemmas'].apply(lambda x: " ".join(x))
    data['without_stop'] = data['lemmas'].apply(remove_stop)
    data['filtered'] = data['tokens'].apply(
        lambda x: filter_pos(x, wanted_pos))
    data['filtered'] = data['filtered'].apply(remove_punc)
    data['filtered'] = data['filtered'].apply(lambda x: ", ".join(x))
    data.drop(['tokens', 'lemmas', 'lemmas_string',
               'without_stop'], axis=1, inplace=True)

    return data


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
    # get get stop words TODO
    stop_words = stopwords.words('danish')
    return [word for word in words if word not in stop_words]


def filter_pos(token, wanted_pos):
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
