'''
Grupo 19
70577 - João Godinho
70643 - João Ferreira
'''
import os
import re
import sys
import string

OUTPATH='normalized'


def normalize(content):
    punct = string.punctuation
    # Keep words with one hifen
    punct = punct.replace('-', '')
    # Keep words with '
    punct = punct.replace('\'', '')

    regex = re.compile(r'[{}]+'.format(re.escape(punct)))
    regex2= re.compile(r'[-]{2,}')
    regex3= re.compile(r'  ')
    regex4= re.compile(r' \n')

    content = re.sub(regex, lambda x: ' ' + x.group(0) + ' ', content)
    content = re.sub(regex2, lambda x: ' ' + x.group(0) + ' ', content)
    content = re.sub(regex3, lambda x: ' ', content)
    content = re.sub(regex4, lambda x: '\n', content)
    return content


if __name__ == '__main__':
    assert len(sys.argv) > 2
    outfile = sys.argv[1]
    files = sys.argv[2:]

    content = ''
    for file in files:
        with open(file, 'r', encoding='UTF-8') as f:
            content += f.read()

    with open(outfile, 'w', encoding='UTF-8') as f:
        f.write(normalize(content))

