'''
'''

import typing
import sentspace.utils as utils


class Sentence:
    '''
    a class to keep track of an individual sentence, its tokenized form,
    lemmas, POS tags, and so on.
    contains handy methods to perform the above segmentation/processing
    operations as well as methods for string representation, equality, and
    indexing (on tokenized form)
    '''
    _uid: str = None
    _raw: str = None
    _tokens: tuple = None
    _pos: tuple = None
    _lemmas: tuple = None
    _content: tuple = None
    
    def __init__(self, raw: str, uid: str = None, warn: bool = True) -> None:
        """Sentence constructor

        Args:
            raw (str): sentence in contiguous string form (will be segmented using a tokenizer)
            uid (str, optional): Unique ID of this sentence in a corpus. Defaults to None.
            warn (bool, optional): whether to warn that a UID is not supplied if one isn't given.
                                   can be set to False to suppress warning in case of intentional
                                   UID-less usage
        """        
        self._raw = raw
        if uid is None and warn:
            utils.io.log(f'no UID supplied for sentence {raw}\r', type='WARN')
        self._uid = uid

    def __eq__(self, other) -> bool:
        """Equality operation with other sentences. Simply compares raw string.

        Args:
            other ([type]): [description]

        Returns:
            bool: True if self is equal to other, else False
        """        
        return str(self) == str(other)

    def __repr__(self) -> str:
        """Implement representation repr() of Sentence.
            prints out the UID followed by sentence

        Returns:
            str: representation of self
        """        
        return f'<{self._uid}>\t {self._raw:<32}'
    def __str__(self) -> str:
        """Returns raw string representation

        Returns:
            str: raw string
        """        
        return f'{self._raw}'

    def __len__(self) -> int:
        """Compute length of the sentence in terms of # of tokens

        Returns:
            int: length of the tokenized sentence
        """        
        return len(self.tokenized())

    def __getitem__(self, index: int) -> str:
        """Support indexing into the tokens of the sentence

        Args:
            index (int): index'th token to retrieve

        Returns:
            str: token at the indexed position in the tokenized form
        """        
        return self.tokenized()[index]

    def tokenized(self, tokenize_method=utils.text.tokenize) -> typing.Tuple[str]:
        """Tokenize and store tokenized form as a tuple. The tokenize_method is executed only
            the first time; each subsequent call to tokenized() returns the value of a stored variable

        Args:
            tokenize_method ([function], optional): a custom tokenization function. Defaults to utils.text.tokenize.

        Returns:
            tuple: tokenized form
        """        
        if self._tokens is None:
            self._tokens = tuple(tokenize_method(self._raw))
        return self._tokens

    def pos_tagged(self) -> typing.Tuple[str]:
        """POS-tag the sentence and return the result. Note, this method is also executed
            only once ever; each subsequent time a stored value is returned.

        Returns:
            tuple: POS tags
        """        
        if self._pos is None:
            self._pos = utils.text.get_pos_tags(self.tokenized())
        return self._pos

    def lemmatized(self) -> typing.Tuple[str]:
        """Lemmatize and return the lemmas

        Returns:
            tuple: Lemmas
        """        
        if self._lemmas is None:
            self._lemmas = utils.text.get_lemmatized_tokens(self.tokenized(), utils.text.get_pos_tags(self.tokenized()))
        return self._lemmas

    def content_words(self) -> typing.List[int]:
        """Check whether each word is a 'content word'. Returns a list containing 0 or 1 as boolean
            mask indicating whether the word at that position is a content word or not

        Returns:
            List[int]: boolean mask indicating content words
        """        
        if self._content is None:
            self._content = utils.text.get_is_content(self.pos_tagged(), content_pos=utils.text.pos_for_content)
        return self._content
