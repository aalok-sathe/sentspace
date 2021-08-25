#!/bin/bash
if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi


# pristine original
echo $input | ./parser-fullberk  wsj02to21.gcg15.prtrm.4sm.fullberk.model| perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'

# modified version
# echo $input | java -Xmx8g -cp berkeleyparser/berkeleyParser.jar edu.berkeley.nlp.PCFGLA.BerkeleyParser -substates -gr wsj02to21.gcg15.prtrm.4sm.fullberk.model -c $1 | perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'