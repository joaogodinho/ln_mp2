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
TOKEN_REGEX = r'\b\w+\'?\w*-?\w*\b'



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
    generate_grams(train)
    test = []
    for i in range(6):
        with open(PATH2 + '/500Palavras/text{}.txt'.format(i+1), 'r', encoding='UTF-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            temp = '\n'.join(list(filter(lambda x: x is not '', temp)))
            temp = temp.replace('_', '')
            test += [temp]
    for i in range(6):
        with open(PATH2 + '/1000Palavras/text{}.txt'.format(i+1), 'r', encoding='UTF-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            temp = '\n'.join(list(filter(lambda x: x is not '', temp)))
            temp = temp.replace('_', '')
            test += [temp]

    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 2))
    dtm = cv.fit_transform(train + test)
    dtm = dtm.toarray()
    dist = np.round(1 - cosine_similarity(dtm), 2)

    # Generate table header
    header = '\t'
    for a in AUTHORS:
        header += '{}\t'.format(a[:5])
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    # Generate table
    table = ''
    for idx, row in enumerate(dist):
        if idx < 6:
            table += '{}\t'.format(AUTHORS[idx][:5])
        else:
            table += 'text{}\t'.format((idx % 6) + 1)
        lowest = min(row)
        for item in row:
            table += '{:.2f}\t'.format(item)
        table += '\n'

    # Print table
    print('(uni/bi)grams all texts:')
    print(header)
    print(table)


    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 1))
    dtm = cv.fit_transform(train + test)
    dtm = dtm.toarray()
    dist = np.round(1 - cosine_similarity(dtm), 2)

    # Generate table header
    header = '\t'
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    # Generate table
    table = ''
    for idx, row in enumerate(dist[:6]):
        table += '{}\t'.format(AUTHORS[idx][:5])
        for item in row[6:]:
            table += '{:.2f}\t'.format(item)
        table += '\n'

    # Print table
    print('Unigrams:')
    print(header)
    print(table)


    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 2))
    dtm = cv.fit_transform(train + test)
    dtm = dtm.toarray()
    dist = np.round(1 - cosine_similarity(dtm), 2)

    # Generate table header
    header = '\t'
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    # Generate table
    table = ''
    for idx, row in enumerate(dist[:6]):
        table += '{}\t'.format(AUTHORS[idx][:5])
        for item in row[6:]:
            table += '{:.2f}\t'.format(item)
        table += '\n'

    # Print table
    print('Bigrams:')
    print(header)
    print(table)


    # cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 1), lowercase=False)
    # dtm = cv.fit_transform(train + test)
    # dtm = dtm.toarray()
    # dist = np.round(1 - cosine_similarity(dtm), 2)

    # # Generate table header
    # header = '\t'
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # # Generate table
    # table = ''
    # for idx, row in enumerate(dist[:6]):
    #     table += '{}\t'.format(AUTHORS[idx][:5])
    #     for item in row[6:]:
    #         table += '{:.2f}\t'.format(item)
    #     table += '\n'

    # # Print table
    # print('Unigrams with lowercase:')
    # print(header)
    # print(table)


    # cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(2, 2), lowercase=False)
    # dtm = cv.fit_transform(train + test)
    # dtm = dtm.toarray()
    # dist = np.round(1 - cosine_similarity(dtm), 2)

    # # Generate table header
    # header = '\t'
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # # Generate table
    # table = ''
    # for idx, row in enumerate(dist[:6]):
    #     table += '{}\t'.format(AUTHORS[idx][:5])
    #     for item in row[6:]:
    #         table += '{:.2f}\t'.format(item)
    #     table += '\n'

    # # Print table
    # print('Bigrams with lowercase:')
    # print(header)
    # print(table)


    stop_words = ['a', 'as', 'com', 'como', 'da', 'de', 'do', 'e', 'em', 'no', 'não', 'o', 'os', 'para', 'por', 'que', 'se', 'um', 'uma', 'é']
    cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(1, 1), stop_words=stop_words)
    dtm = cv.fit_transform(train + test)
    dtm = dtm.toarray()
    dist = np.round(1 - cosine_similarity(dtm), 2)

    # Generate table header
    header = '\t'
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    for i in range(6):
        header += 'text{}\t'.format(i+1)
    # Generate table
    table = ''
    for idx, row in enumerate(dist[:6]):
        table += '{}\t'.format(AUTHORS[idx][:5])
        for item in row[6:]:
            table += '{:.2f}\t'.format(item)
        table += '\n'

    # Print table
    print('Unigrams with top 20 stop words ({}):'.format(stop_words))
    print(header)
    print(table)


    # stop_words = ['a sua', 'com a', 'com o', 'como se', 'de que', 'de um', 'e a', 'e o', 'em que', 'não é', 'o que', 'o seu', 'os olhos', 'para a', 'para o', 'que a', 'que não', 'que o', 'que se', 'é que']
    # cv = CountVectorizer(token_pattern=TOKEN_REGEX, ngram_range=(2, 2), stop_words=stop_words)
    # dtm = cv.fit_transform(train + test)
    # dtm = dtm.toarray()
    # dist = np.round(1 - cosine_similarity(dtm), 2)

    # # Generate table header
    # header = '\t'
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # for i in range(6):
    #     header += 'text{}\t'.format(i+1)
    # # Generate table
    # table = ''
    # for idx, row in enumerate(dist[:6]):
    #     table += '{}\t'.format(AUTHORS[idx][:5])
    #     for item in row[6:]:
    #         table += '{:.2f}\t'.format(item)
    #     table += '\n'

    # # Print table
    # print('Bigrams with top 20 stop words ({}):'.format(stop_words))
    # print(header)
    # print(table)
