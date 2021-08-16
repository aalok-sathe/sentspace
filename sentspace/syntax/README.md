van Schijndel et al. (2013) parser implementation
=================================================

This directory contains scripts to run an implementation of the parser
described in van Schijndel et al. (2013), pre-trained on sections 2-21
of the Wall Street Journal corpus of the Penn Treebank (Marcus et al.,
1993). This is the pre-trained model that was used to compute PCFG
surprisal values investigated in Shain, Blank, et al. (2020).

The system produces a table of decisions that describe the 1-best left-
corner parse of each input sentence, along with word surprisal
computed by marginalizing over all parses in the (5000-dimensional)
beam at each timestep.

All scripts assume that this directory is your working directory. In
addition, all scripts write to standard out, and will need to be
redirected to files.

Input Format
------------

The system requires textual input, one sentence per line, space-
tokenized following the Penn Treebank standard (with punctuation
and contractions separated out as separate tokens), like so:

This is a sentence .
Is n't this story , actually , very interesting ?

Because parses are represented using parentheses for bracketing,
it's also usually a good idea to convert parentheses in the input
to textual tokens, commonly `-LRB-` for left paren and `-RRB-` for
right paren. If you don't do this and your input has parentheses,
your output trees may not be machine-readable (because of unmatched
brackets).

For convenience, this directory contains the script `tokenize.sh`,
which automatically tokenizes raw text as above, and also normalizes
punctuation (e.g. parentheses). To use it on text file `example_raw.txt`
and redirect the output to file `example.txt`, run:
```
./tokenize.sh example_raw.txt > example.txt
```

Parsing
-------

Parsing serves two purposes in this pipeline: generating word-by-word
PCFG surprisal measures, and generating parse trees that can be used
to extract syntactic features. A different parser is used for each of
these purposes: synproc is (van Schijndel et al, 2013) is required
for surprisal, but a faster and more accurate chart parser is used
to compute trees.

In the examples below, the filenames `example.txt`, `output.txt`,
`surp.txt`, and `trees.txt` can (and should) be changed to reflect
your use case.

PARSING FOR SURPRISAL:

Given input `example.txt`, and a desired output file `output.txt`, 
run the parser as follows:

```
./parse_surp.sh example.txt > output.txt
```

This will generate a space-delimited table of parse decisions that
are not very human-readable. One column of the table, `totsurp`, 
contains PCFG surprisal values. This table can be used directly,
or, for convenience, you can do the following, which just deletes
columns irrelevant to surprisal:

```
./get_surp.sh output.txt > surp.txt
```

If you want to see the trees, run:

```
./get_trees.sh output.txt > trees.txt
```

PARSING FOR TREES:

Given input `example.txt`, and a desired output file `output.txt`, 
run the parser as follows:

```
./parse_trees.sh example.txt > output.txt
```

This will generate a forest of maximum likelihood parse trees,
one per line. Note that there is substantial initial overhead
from loading the model into memory.

DLT features
------------

Given GCG-15 trees (one per line) `example_trees.txt` generated either
via hand annotation or automatic parsing (see PARSING FOR TREES above),
DLT features can be extracted as follows:

```
./dlt.sh example_trees.txt > output.txt
```

This will generate a space-delimited table of DLT features, one row
per token. The key columns of interest are the following:

- dlt: DLT integration cost per Gibson (2000)
- dlt{,c}{,v}{,m}: DLT integration cost with respective combination
of C, V, and M modifications (see Shain et al., 2016, for details)
- dlts: DLT storage cost

Left-corner features
--------------------

Given GCG-15 trees (one per line) `example_trees.txt` generated either
via hand annotation or automatic parsing (see PARSING FOR TREES above),
left-corner features can be extracted as follows:

```
./leftcorner.sh example_trees.txt > output.txt
```

This will generate a space-delimited table of left-corner features, one
row per token. The key columns of interest are the following (for details,
see Rasmussen & Schuler, 2018):

- noF: End of constituent
- noFlen: Length of consituent (in words)
- noFdr: Length of constituent (in DLT-style discourse referents)
- noFdrv: Length of constituent (in DLT-style discourse reference using -V modification)
- yesJ: End of center-embedding
- startembdMin: Start of multiword center-embedding
- endembdMin: End of multiword center-embedding
- embdlen: Length of multiword center-embedding (in words)
- embddr: Length of multiword center-embedding (in DLT-style discourse referents)
- embddrv: Length of multiword center-embedding (in DLT-style discourse referents using -V modification)
- embddepthMin: Memory stack depth

Citations
---------

Relevant citations for these resources are given in the
`citations.bib` file in this directory.


