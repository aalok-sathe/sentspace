#!/bin/bash

if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi

echo $1 | perl ptb_tokenizer.sed | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;s/ $//g;'
