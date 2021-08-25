import os
from pathlib import Path

from sentspace.utils import text, wordnet


def path_decorator(func):
    """Decorator that changes to and from the directory containing scripts
        for running DLT and Left Corner metrics before and after a function
        call respectively.
    """

    def wrapped(*args, **kwargs):
        ''' function that changes the directory to an expected directory,
            executes original function with supplied args, and changes
            back to the same directory we started from
        '''
        previous_pwd = os.getcwd()
        target = Path(__file__)
        os.chdir(str(target.parent))
        result = func(*args, **kwargs)
        os.chdir(previous_pwd)
        return result.decode('utf-8').strip()

    return wrapped



def get_content_ratio(is_content_pos_tag: tuple):
    """
    given boolean list corresponding to a token being a content word, 
    calculate the content ratio
    """
    return sum(is_content_pos_tag) / len(is_content_pos_tag)


def get_pronoun_ratio(pos_tags: tuple):
    """
    Given sentence calculate the pronoun ratio
    """
    pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}
    return sum(tag in pronoun_tags for tag in pos_tags) / len(pos_tags)


# @cache_to_mem
def get_is_content(taglst: tuple, content_pos=(wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV)):
    """
    Given list of POS tags, return list of 1 - content word, 0 - not content word
    """
    return tuple(int(text.get_wordnet_pos(tag) in content_pos) for tag in taglst)
