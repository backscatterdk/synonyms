"""
This module takes a csv with a column of lists of terms as input and changes terms that similar.
"""

# -*- coding: utf-8 -*-

import json
import shutil
import itertools
import csv
import os
import argparse
import numpy as np
import pandas as pd
import click
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from pick import pick
from tqdm import tqdm
from googletrans import Translator
from ratelimit import limits, sleep_and_retry
from pyjarowinkler import distance


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
    cutoff_msg = f'You have {len(list(set(terms)))} terms. Please set \
a minimum number of occurences (or Enter to use all terms)'
    cutoff = ask_cutoff(cutoff_msg, 0)
    count = pd.Series(terms).value_counts()
    # print(count)
    unique_terms = list(count[count > cutoff].index)
    msg = f'This gives you {len(unique_terms)} terms. Continue? \
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


@sleep_and_retry
@limits(calls=15)  # This limits calls to 1 a minute.
def make_translations(lst):
    """
    This function takes in a list of words, and then detects the languages of each and
    translates them if they are not English. It uses the googletrans api.
    """
    print('\nTranslating...')
    out = []
    too_many_requests = False
    for word in tqdm(lst):
        if not too_many_requests:
            try:
                translator = Translator()
                lang = translator.detect(word).lang
                if not 'en' in lang:
                    translated = translator.translate(word, dest='en').text
                    out.append(translated)
                else:
                    out.append(word)
            except json.decoder.JSONDecodeError:
                print('You have met the request limit. \
Will continue without translating.')
                out.append(word)
                too_many_requests = True
        else:
            out.append(word)
    return out


def find_distances():
    """
    Main function for calculating distance on whole df.
    """
    cutoff_msg = "At what distance would you like to cut off \
word similarities? Default: "
    cutoff = ask_cutoff(cutoff_msg, 0.8)

    # Read combinations csv and convert dask df
    df = pd.read_csv('combinations.csv', dtype='str')
    ddf = dd.from_pandas(df, npartitions=12)

    # Calculate lev distance
    ddf['lev'] = ddf.apply(lev_similarity, meta=float, axis=1)
    print('\nCalculating distances...')
    with ProgressBar():
        ddf.persist()

    # Remove below cutoff
    ddf = ddf[ddf.lev > cutoff]
    print('\nRemoving pairs under cutoff...')
    with ProgressBar():
        ddf.persist()

    # Write to seperate dask csv's (default way dask does it)
    print('\nWriting files...')
    outname = 'lev-*.csv'
    outdir = os.path.join(os.getcwd(), 'levs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    ddf.to_csv(fullname)

    # Append seperate csv's to one file
    from glob import glob
    filenames = glob(fullname)
    with open('main_lev.csv', 'w') as out:
        for file in tqdm(filenames):
            with open(file) as f:
                out.write(f.read())

    # Delete old seperate csv's
    if os.path.exists(outdir):
        shutil.rmtree(outdir)


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


def lev_similarity(row):
    """
    Returns distance between two terms.
    """
    try:
        return distance.get_jaro_distance(
            row[0],
            row[1],
            winkler=True,
            scaling=0.1)
    except:
        return np.nan


PARSER = argparse.ArgumentParser()

PARSER.add_argument("data", help="Path to your data file.")

ARGS = PARSER.parse_args()

DATA = pd.read_csv(ARGS.DATA)

TERMS = find_terms(DATA)

# Translate
TRANS_TITLE = 'Would like to translate everything to English?'
USE_TRANS = ['yes', 'no']
USE_TRANS, TRANS_TITLE = pick(USE_TRANS, TRANS_TITLE)
if USE_TRANS == 'yes':
    TERMS = make_translations(TERMS)

with open('log.txt', 'w') as f:
    for l in TERMS:
        f.write(l)
        f.write('\n')

# Write csv with combinations (too big for memory)
with open('combinations.csv', 'w', newline='') as outfile:
    OUTWRITER = csv.writer(outfile,
                           delimiter=',',
                           quotechar='"',
                           quoting=csv.QUOTE_MINIMAL)
    print('\nMaking pairs of all terms.')
    for combo in tqdm(itertools.combinations(TERMS, 2)):
        OUTWRITER.writerow(combo)

# Distance
LEV_TITLE = 'Would like to use word distance?'
USE_LEV = ['yes', 'no']
USE_LEV, LEV_TITLE = pick(USE_LEV, LEV_TITLE)
if USE_LEV == 'yes':
    find_distances()
