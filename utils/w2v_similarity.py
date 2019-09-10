"""
Functions related to finding Word2Vec similarity.
"""

import os
import shutil
import logging
import multiprocessing
from glob import glob
import urllib.request
from gensim.models.word2vec import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """
    Makes a progress bar for downloading w2v data
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def find_w2v(cutoff):
    """
    Main word 2 vec function
    """

    # Set up logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Read combinations csv as dask df
    ddf = dd.read_csv('combinations.csv', dtype='str')

    # For downloading articles from wikipedia
    url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
    articles_folder = os.path.join(os.getcwd(), 'w2v_files')
    if not os.path.exists(articles_folder):
        os.mkdir(articles_folder)

    # Split on the rightmost / and take everything on the right side of that
    articles_file = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    articles_path = os.path.join(articles_folder, articles_file)

    # Download the file if it does not exist
    if not os.path.isfile(articles_path):
        download_url(url, articles_path)

    # For training model
    model_path = os.path.join(articles_folder, 'wiki.model')
    if not os.path.isfile(model_path):
        model = train_and_save_model(articles_path, model_path)

    # Apply model
    ddf['w2v'] = ddf.apply(w2v_similarity, args=(model), extra_kw=1, axis=1)
    with ProgressBar():
        ddf.persist()

    # Remove below cutoff
    ddf = ddf[ddf.lev > cutoff]
    print('\nRemoving pairs under cutoff...')
    with ProgressBar():
        ddf.persist()

    # Write to seperate dask csv's (default way dask does it)
    print('\nWriting files...')
    outname = 'w2v-*.csv'
    outdir = os.path.join(os.getcwd(), 'w2vs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    ddf.to_csv(fullname)

    # Append seperate csv's to one file
    from glob import glob
    filenames = glob(fullname)
    with open('main_w2v.csv', 'w') as out:
        for file in tqdm(filenames):
            with open(file) as f:
                out.write(f.read())

    # Delete wiki data
    if os.path.isfile(articles_path):
        os.remove(articles_path)

    # Delete old seperate csv's
    if os.path.exists(outdir):
        shutil.rmtree(outdir)


def download_url(url, output_path):
    """Download data for making w2v model"""
    print('Beginning file download with urllib2...')
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def train_and_save_model(articles_path, model_path):
    corpus = WikiCorpus(articles_path, lemmatize=False, dictionary={})
    sentences = list(corpus.get_texts())
    params = {'size': 200, 'window': 10, 'min_count': 10,
              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3, }
    model = Word2Vec(sentences, **params)
    model.save(model_path)

    return model


def w2v_similarity(row, model):
    """
    Test w2v-similarity between two words
    """
    try:
        return model.similarity(row[0], row[1])
    except:
        return np.nan


def find_w2v(cutoff):
    """
    Main word 2 vec function
    """

    # Set up logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Read combinations csv as dask df
    ddf = dd.read_csv('combinations.csv', dtype='str')

    # For downloading articles from wikipedia
    url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
    articles_folder = os.path.join(os.environ['HOME'], 'word2vec_resources')
    if not os.path.exists(articles_folder):
        os.mkdir(articles_folder)

    # Split on the rightmost / and take everything on the right side of that
    articles_file = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    articles_path = os.path.join(articles_folder, articles_file)

    # Download the file if it does not exist
    if not os.path.isfile(articles_path):
        download_url(url, articles_path)

    # For training model
    model_path = os.path.join(articles_folder, 'wiki.model')
    if not os.path.isfile(model_path):
        model = train_and_save_model(articles_path, model_path)

    # Apply model
    ddf['w2v'] = ddf.apply(w2v_similarity, args=(model), extra_kw=1, axis=1)
    with ProgressBar():
        ddf.persist()

    # Remove below cutoff
    ddf = ddf[ddf.lev > cutoff]
    print('\nRemoving pairs under cutoff...')
    with ProgressBar():
        ddf.persist()

    # Write to seperate dask csv's (default way dask does it)
    print('\nWriting files...')
    outname = 'w2v-*.csv'
    outdir = os.path.join(os.getcwd(), 'w2vs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    ddf.to_csv(fullname)

    # Append seperate csv's to one file
    filenames = glob(fullname)
    with open('main_w2v.csv', 'w') as out:
        for subfile in tqdm(filenames):
            with open(subfile) as f:
                out.write(f.read())

    # Delete wiki data
    if os.path.isfile(articles_path):
        os.remove(articles_path)

    # Delete old seperate csv's
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
