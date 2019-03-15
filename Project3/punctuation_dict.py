import os
import re

def get_punctuation_dict():
    '''
    Takes in no input and returns a dictionary with keys as unpunctuated words
    and values as corresponding punctuated words (original words from
    Syllables_dictionary.txt).
    '''
    file = open(os.path.join(os.getcwd(),
                'data/Syllable_dictionary.txt')).read()
    lines = [line.split() for line in file.split('\n') if line.split()]

    punctuation_dict = {}

    for line in lines:
        word = line[0]
        punctuation_dict[re.sub(r'[^\w]', '', word)] = word

    return punctuation_dict
