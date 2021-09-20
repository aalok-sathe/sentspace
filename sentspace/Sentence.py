'''
'''


import sentspace.utils


class Sentence:
    _raw: str = None
    _tokens = None
    
    def __init__(self) -> None:
        super().__init__()

    def tokenized(self, tokenize_method=sentspace.utils.text.tokenize):
        if self._tokens is None:
            self._tokens = tokenize_method()
