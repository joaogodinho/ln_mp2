#!/bin/bash
# Grupo 19
# 70577 - João Godinho
# 70643 - João Ferreira

TRAIN='Corpora/treino'
VALIDATION='Corpora/teste'
OUTNORM='normalized'

function init {
    rm -rf normalized ngrams
    mkdir -p normalized/train
    mkdir -p normalized/test/500Palavras
    mkdir -p normalized/test/1000Palavras
    mkdir -p ngrams
    for f in $TRAIN/*
    do
        mkdir -p ngrams/$(basename $f)
    done
}

function normalize {
    for f in $TRAIN/*
    do
        NAME=$(basename $f)
        OUTPUT="$OUTNORM/train/$NAME.txt"
        echo "Normalizing $NAME..."
        python normalize.py $OUTPUT $f/*
    done
    for f in $VALIDATION/*
    do
        NAME=$(basename $f)
        echo "Normalizing $NAME..."
        for txt in $f/*
        do
            NAME="$NAME/$(basename $txt)"
            OUTPUT="$OUTNORM/test/$NAME"
            python normalize.py $OUTPUT $txt
            NAME=$(basename $f)
        done
    done
}


init
normalize
python ngrams.py
