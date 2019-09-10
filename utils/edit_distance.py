"""
Functions related to finding edit distance. Currently uses Jaro-Winkler.
"""


import shutil
import os
from glob import glob
from pyjarowinkler import distance
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
from tqdm import tqdm
import numpy as np


def find_distances(cutoff):
    """
    Main function for calculating distance on whole df.
    """

    # Read combinations csv as dask df
    ddf = dd.read_csv('combinations.csv', dtype='str')

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
    filenames = glob(fullname)
    with open('main_lev.csv', 'w') as out:
        for file in tqdm(filenames):
            with open(file) as f:
                out.write(f.read())

    # Delete old seperate csv's
    if os.path.exists(outdir):
        shutil.rmtree(outdir)


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
