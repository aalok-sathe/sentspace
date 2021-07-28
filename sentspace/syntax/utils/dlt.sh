#!/bin/bash

if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi

echo $input | python3 calcDLT.py
