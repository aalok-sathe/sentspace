#!/bin/bash

if [[ -f $1 ]]; then
    input=$(cat $1)
else
    input=$1
fi

linetrees=$( echo $input  |  PERL_BADLANG=0 perl make-trees-nounary.pl |  perl -pe "s/ \([^ ()]+ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;)\)//g" | PERL_BADLANG=0 perl make-trees-nounary.pl )
lemmacounts=$(  echo $linetrees  |  python2.7 printlemmas.py  |  LC_COLLATE=C sort  |  LC_COLLATE=C uniq -c  |  sed 's/^ *//' )
echo $linetrees  |  ./linetrees2mlpsemprocdecpars -u0 -c0 $lemmacounts > decpars.tmp
echo $input | sed 's/(-NONE-[^)]*)//g' |  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$//;'  |  sed 's/!unf! *//g' | python2.7 calcEmbd.py decpars.tmp
rm decpars.tmp
