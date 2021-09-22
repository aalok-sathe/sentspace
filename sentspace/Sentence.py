'''
'''

import typing
import sentspace.utils as utils


class Sentence:
    _uid: str = None
    _raw: str = None
    _tokens: tuple = None
    _pos: tuple = None
    _lemmas: tuple = None
    _content: tuple = None
    
    def __init__(self, raw: str, uid: str = None) -> None:
        self._raw = raw
        if uid is None:
            utils.io.log(f'no UID supplied for sentence {raw}\r', type='WARN')
        self._uid = uid

    def __eq__(self, other) -> bool:
        return self._raw == other._raw

    def __repr__(self) -> str:
        return f'<{self._uid:.>7}>\t {self._raw:<32}'
    def __str__(self) -> str:
        return f'{self._raw}'

    def __len__(self) -> int:
        return len(self.tokenized())

    def __getitem__(self, index: int) -> str:
        return self.tokenized()[index]

    def tokenized(self, tokenize_method=utils.text.tokenize) -> tuple:
        if self._tokens is None:
            self._tokens = tuple(tokenize_method(self._raw))
        return self._tokens

    def pos_tagged(self) -> tuple:
        if self._pos is None:
            self._pos = utils.text.get_pos_tags(self.tokenized())
        return self._pos

    def lemmatized(self):
        if self._lemmas is None:
            self._lemmas = utils.text.get_lemmatized_tokens(self.tokenized(), utils.text.get_pos_tags(self.tokenized()))
        return self._lemmas

    def content_words(self):
        if self._content is None:
            self._content = utils.text.get_is_content(self.pos_tagged(), content_pos=utils.text.pos_for_content)
        return self._content