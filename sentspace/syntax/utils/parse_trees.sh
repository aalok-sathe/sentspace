#!/bin/bash
if [[ -f $2 ]]; then
    input=$(cat $2)
else
    input=$2
fi

# echo $(date) "beam_width param received -c $1 $2" >> parse_trees.log

# pristine original
echo $input | ./parser-fullberk  wsj02to21.gcg15.prtrm.4sm.fullberk.model| perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'
# modified version
# echo $input | java -Xmx8g -cp berkeleyparser/berkeleyParser.jar edu.berkeley.nlp.PCFGLA.BerkeleyParser -substates -gr wsj02to21.gcg15.prtrm.4sm.fullberk.model -c $1 | perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$//;s/-\d+ / /g;s/-PRTRM//g'
