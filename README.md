# Synonyms
This module lets you find word synonyms through both a word2vector and edit distance method. It also uses the Google Translate API to translate words to English. The module itself downloads the English Wikipedia dataset of most recent articles and trains a word2vec-model based on that using [gensim](https://pypi.org/project/gensim/). Edit distance is measured through the [Jaro-Winkler metric](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance). The module itself is set up as a CLI application, which you can run by simply cloning this library and running the following in the download folder, providing the path to your data file:

`python synonyms data.csv`

Your data should have a column with the words you wish to find synonyms for. They can have a single word per cell, or be separated with a delimiter such as `,` or `;`.

| id | col1 | col2 | col3                          |
|----|------|------|-------------------------------|
| 1  |      |      | lorem,ipsum,dolor,sit      |
| 2  |      |      | amet,consectetur,adipiscing |
| 3  |      |      | pellentesque                  |



