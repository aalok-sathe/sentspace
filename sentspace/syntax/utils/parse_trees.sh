#!/bin/bash
if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi

echo $input | ./parser-fullberk  wsj02to21.gcg15.prtrm.4sm.fullberk.model | perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'

