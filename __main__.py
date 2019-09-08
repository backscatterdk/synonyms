"""
This module takes a csv with a column of lists of terms
as input and changes terms that similar.
"""

# -*- coding: utf-8 -*-


import itertools
import csv
import argparse
import numpy as np
import pandas as pd
import click
from pick import pick
from tqdm import tqdm
import utils.w2v_similarity as w2v
import utils.edit_distance as ed
import utils.translations as trans


def find_terms(df):
    """
    Returns a list of all unique terms
    """
    # Pick column for terms
    column_title = 'Please chose which column contains the words.'
    columns = df.columns
    column, column_title = pick(columns, column_title)

    # Seperator?
    sep = input("What seperator is used?")

    # First split user-lists of terms into python-lists
    all_terms = df[column].apply(lambda cell: split_lists(cell, sep))

    # One big list
    terms = [term for sublist in all_terms for term in sublist]

    unique_terms = min_occurence(terms)

    return unique_terms


def min_occurence(terms):
    """
    Finds minimum occurences.
    """
    cutoff_msg = f'\nYou have {len(list(set(terms)))} terms. Please set \
a minimum number of occurences (or Enter to use all terms)'
    cutoff = ask_cutoff(cutoff_msg, 0)
    count = pd.Series(terms).value_counts()
    # print(count)
    unique_terms = list(count[count > cutoff].index)
    msg = f'\nThis gives you {len(unique_terms)} terms. Continue? \
n or Enter to continue'
    do_again = click.prompt(text=msg, default='y')
    if do_again == 'y':
        return unique_terms
    if do_again == 'n':
        min_occurence(terms)
    else:
        print('\nPlease respond with y or n. Asking again...')
        min_occurence(terms)


def split_lists(cell, sep):
    """
    Splits cell of words seperated by 'sep' into python lists.
    """
    if not pd.isnull(cell):
        return cell.split(sep)
    return [np.nan]


def ask_cutoff(msg, default):
    """
    This simply provides a secure wrapper for checking user input.
    """
    try:
        cutoff = click.prompt(msg, default=default)
        return cutoff
    except ValueError:
        print("\nPlease specify an integer")
        ask_cutoff(msg, default)


PARSER = argparse.ArgumentParser()

PARSER.add_argument("data", help="Path to your data file.")

ARGS = PARSER.parse_args()

DATA = pd.read_csv(ARGS.data)

TERMS = find_terms(DATA)

# Translate
TRANS_TITLE = 'Would like to translate everything to English?'
USE_TRANS = ['yes', 'no']
USE_TRANS, TRANS_TITLE = pick(USE_TRANS, TRANS_TITLE)
if USE_TRANS == 'yes':
    TERMS = trans.make_translations(TERMS)

# Write csv with combinations (too big for memory)
with open('combinations.csv', 'w', newline='') as outfile:
    OUTWRITER = csv.writer(outfile,
                           delimiter=',',
                           quotechar='"',
                           quoting=csv.QUOTE_MINIMAL)
    print('\nMaking pairs of all terms...')
    TQDM_TOTAL = len(TERMS)**2
    for combo in tqdm(itertools.combinations(TERMS, 2), total=TQDM_TOTAL):
        OUTWRITER.writerow(combo)

# Distance
LEV_TITLE = 'Would like to use word distance?'
USE_LEV = ['yes', 'no']
USE_LEV, LEV_TITLE = pick(USE_LEV, LEV_TITLE)
if USE_LEV == 'yes':
    ed.find_distances()

# w2v
W2V_TITLE = 'Would like to use word2vec?'
USE_W2V = ['yes', 'no']
USE_W2V, W2V_TITLE = pick(USE_W2V, W2V_TITLE)
if USE_W2V == 'yes':
    w2v.find_w2v()
