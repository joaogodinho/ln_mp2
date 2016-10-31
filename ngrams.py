'''
Grupo 19
70577 - João Godinho
70643 - João Ferreira
'''
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


PATH = 'Corpora'


def unigram(content):
    cv = CountVectorizer(token_pattern=r'\b\w+-?\w*\b', min_df=0)

    dtm = cv.fit_transform(content)
    dtm = dtm.toarray()
    vocab = cv.get_feature_names()

    dist = np.sum(dtm, axis=0)
    content = ''
    for tag, count in zip(vocab, dist):
        content += '{}\t{}\n'.format(tag, count)
    print(content)
    return content


def bigram(content):
    cv = CountVectorizer(token_pattern=r'\b\w+-?\w*\b', ngram_range=(2, 2))

    dtm = cv.fit_transform(content)
    dtm = dtm.toarray()
    vocab = cv.get_feature_names()

    dist = np.sum(dtm, axis=0)
    content = ''
    for tag, count in zip(vocab, dist):
        content += '{}\t{}\n'.format(tag, count)
    return content


# if __name__ == '__main__':
#     assert len(sys.argv) == 3
#     input = sys.argv[1]
#     outdir = sys.argv[2]
# 
#     content = ''
#     with open(input, 'r', encoding='UTF-8') as f:
#         content = f.read()
#     content = content.split('\n')
#     content = list(filter(lambda x: x is not '', content))
#     content = "\n".join(content)
#     content = content.replace('_', '')
#     unigrams = unigram([content])
#     bigrams = bigram([content])
#     with open(outdir + '/unigram.txt', 'w', encoding='UTF-8') as f:
#         f.write(unigrams)
#     with open(outdir + '/bigram.txt', 'w', encoding='UTF-8') as f:
#         f.write(bigrams)

if __name__ == '__main__':
    # assert len(sys.argv) == 3
    #input = sys.argv[1]
    #outdir = sys.argv[2]

    content = []
    for f in sys.argv[1:]:
        with open(f, 'r', encoding='UTF-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            temp = list(filter(lambda x: x is not '', temp))
            temp = "\n".join(temp)
            temp = temp.replace('_', '')
            content += [temp]
    print(unigram(content))
    # with open(input, 'r', encoding='UTF-8') as f:
    #     content = f.read()
    # content = content.split('\n')
    # content = list(filter(lambda x: x is not '', content))
    # content = "\n".join(content)
    # content = content.replace('_', '')
    # unigrams = unigram([content])
    # bigrams = bigram([content])
    # with open(outdir + '/unigram.txt', 'w', encoding='UTF-8') as f:
    #     f.write(unigrams)
    # with open(outdir + '/bigram.txt', 'w', encoding='UTF-8') as f:
    #     f.write(bigrams)
