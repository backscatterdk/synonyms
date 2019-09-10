"""
Functions related to translating words.
"""

import os
import json
from tqdm import tqdm
from googletrans import Translator
from ratelimit import limits, sleep_and_retry


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

    # Make log of translations
    transDict = {}

    # Translate words
    for word in tqdm(lst):
        # Check if api-limit is reached
        if not too_many_requests:
            try:
                # Detect word
                translator = Translator()
                lang = translator.detect(word).lang
                # If not English, translate, append to output, and log
                if not 'en' in lang:
                    translated = translator.translate(word, dest='en').text
                    out.append(translated)
                    transDict.update({word: translated})
                else:
                    # Only translate if English
                    out.append(word)
            except json.decoder.JSONDecodeError:
                print('You have met the request limit. \
Will continue without translating.')
                out.append(word)
                too_many_requests = True
        else:
            out.append(word)

    # Save log
    transLog = 'translation_log.txt'
    if os.path.isfile(transLog):
        os.remove(transLog)
    with open(transLog, 'w') as json_file:
        json.dump(transDict, json_file)

    return out
