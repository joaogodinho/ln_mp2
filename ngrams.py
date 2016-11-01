'''
Grupo 19
70577 - João Godinho
70643 - João Ferreira
'''
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PATH = 'normalized/train'
PATH2= 'normalized/test'
AUTHORS = [
        'AlmadaNegreiros',
        'CamiloCasteloBranco',
        'EcaDeQueiros',
        'JoseRodriguesSantos',
        'JoseSaramago',
        'LuisaMarquesSilva'
]
TOKEN_REGEX = r'\b\w+-?\w*\b'



def gram_output(arr, vocab):
    unigrams = ''
    bigrams = ''
    for tag, count in zip(vocab, arr):
        if count > 0:
            if ' ' in tag:
                bigrams += '{}\t{}\n'.format(tag, count)
            else:
                unigrams += '{}\t{}\n'.format(tag, count)
    return unigrams, bigrams


def gram_smooth_output(arr, vocab):
    unigrams = ''
    bigrams = ''
    for tag, count in zip(vocab, arr):
        if ' ' in tag:
            bigrams += '{}\t{}\n'.format(tag, count + 1)
        else:
            unigrams += '{}\t{}\n'.format(tag, count + 1)
    return unigrams, bigrams


def generate_grams(content):
    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 2))
    dtm = cv.fit_transform(train)
    dtm = dtm.toarray()
    vocab = cv.get_feature_names()
    for idx, author in enumerate(AUTHORS):
        unigrams, bigrams = gram_output(dtm[idx], vocab)
        unigrams_smooth, bigrams_smooth = gram_smooth_output(dtm[idx], vocab)
        with open('ngrams/' + author + '/unigrams.txt', 'w', encoding='UTF-8') as f:
            f.write(unigrams)
        with open('ngrams/' + author + '/bigrams.txt', 'w', encoding='UTF-8') as f:
            f.write(bigrams)
        with open('ngrams/' + author + '/unigrams_smooth.txt', 'w', encoding='UTF-8') as f:
            f.write(unigrams_smooth)
        with open('ngrams/' + author + '/bigrams_smooth.txt', 'w', encoding='UTF-8') as f:
            f.write(bigrams_smooth)


if __name__ == '__main__':
    train = []
    for author in AUTHORS:
        with open(PATH + '/' + author + '.txt', 'r', encoding='UTF-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            temp = '\n'.join(list(filter(lambda x: x is not '', temp)))
            temp = temp.replace('_', '')
            train += [temp]
    # Generate the (uni/bi)grams files for evaluation
    # generate_grams(train)
    test = []
    for i in range(6):
        with open(PATH2 + '/500Palavras/text{}.txt'.format(i+1), 'r', encoding='UTF-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            temp = '\n'.join(list(filter(lambda x: x is not '', temp)))
            temp = temp.replace('_', '')
            test += [temp]

    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 1), max_features=100)
    dtm = cv.fit_transform(train + test)
    dtm = dtm.toarray()
    dist = np.round(1 - cosine_similarity(dtm), 2)

    # Generate table header
    header = ''
    for auth in AUTHORS:
        header += auth[:5] + '\t'
    for i in range(6):
        header += 'text{}\t'.format(i+1)

    # Print table
    print(header)
    print('\n'.join([''.join(['{:.2f}\t'.format(item) for item in row]) for row in dist]))

