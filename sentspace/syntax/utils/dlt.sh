#!/bin/bash

if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi

echo $input | python2 calcDLT.py
