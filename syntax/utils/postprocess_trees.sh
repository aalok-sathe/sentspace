#!/bin/bash
input=$1

# newer version to support server-side parsing
echo $input | perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'
