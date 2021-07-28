#!/bin/sed -f

### Original script (1995) available at: http://www.cis.upenn.edu/~treebank/tokenizer.sed
###   by Robert MacIntyre, University of Pennsylvania
### Extended script (forked in 2012) available at: https://github.com/vansky/extended_penn_tokenizer

# Sed script to produce Penn Treebank tokenization on arbitrary raw text.
# expected input: raw text with ONE SENTENCE TOKEN PER LINE

# attempt to get correct directional quotes
s=^"=`` =g
s=\([ ([{<]\)"=\1 `` =g
s=``= `` =g
# close quotes handled at end

s=\.\.\.= ... =g
s=[;:@#$%&]= & =g

#tokenize commas only if they aren't part of numbers
s=,\([^0-9]\)= , \1=g

# Assume sentence tokenization has been done first, so split FINAL periods
# only. 
s=\([^.]\)\([.]\)\([])}>"']*\)[ 	]*$=\1 \2\3 =g
# however, we may as well split ALL question marks and exclamation points,
# since they shouldn't have the abbrev.-marker ambiguity problem
s=[?!]= & =g

# parentheses, brackets, etc.
s=[][(){}<>]= & =g
# Some taggers, such as Adwait Ratnaparkhi's MXPOST, use the parsed-file
# version of these symbols.
# UNCOMMENT THE FOLLOWING 6 LINES if you're using MXPOST.
s/(/-LRB-/g
s/)/-RRB-/g
s/\[/-LSB-/g
s/\]/-RSB-/g
s/{/-LCB-/g
s/}/-RCB-/g

s=--= -- =g

# NOTE THAT SPLIT WORDS ARE NOT MARKED.  Obviously this isn't great, since
# you might someday want to know how the words originally fit together --
# but it's too late to make a better system now, given the millions of
# words we've already done "wrong".

# First off, add a space to the beginning and end of each line, to reduce
# necessary number of regexps.
s=$= =
s=^= =

# capture open quotes
s=\([^nN]\)'\([^('|ll|re|ve|LL|RE|VE|t|T|s|S|d|m)]\)=\1' \2=g
s=\([^nN]\)'\([tT]\)=\1' \2=g
s= '\([lL]\)\([^lL]\)=' \1\2=g
s= '\([rR]\)\([^eE]\)=' \1\2=g
s= '\([vV]\)\([^eE]\)=' \1\2=g

s="= '' =g
# possessive or close-single-quote
s=\([^' ]\)'\([ ']\)=\1 '\2=g
# as in it's, I'm, we'd
s='\([sSmMdD]\) = '\1 =g
s='ll = 'll =g
s='re = 're =g
s='ve = 've =g
s=n't = n't =g
s='d = 'd =g
s='m = 'm =g
s='LL = 'LL =g
s='RE = 'RE =g
s='VE = 'VE =g
s=N'T = N'T =g



s= \([Cc]\)annot = \1an not =g
s= \([Dd]\)'ye = \1' ye =g
s= \([Gg]\)imme = \1im me =g
s= \([Gg]\)onna = \1on na =g
s= \([Gg]\)otta = \1ot ta =g
s= \([Ll]\)emme = \1em me =g
s= \([Mm]\)ore'n = \1ore 'n =g
s= '\([Tt]\)is = '\1 is =g
s= '\([Tt]\)was = '\1 was =g
s= \([Ww]\)anna = \1an na =g
# s= \([Ww]\)haddya = \1ha dd ya =g
# s= \([Ww]\)hatcha = \1ha t cha =g

# clean out extra spaces
s=  *= =g
s=^ *==g
