
import typing
from joblib import Memory
from pathlib import Path
from functools import lru_cache, partial


_cache_instance = Memory(location=Path(__file__).parent / '..' / '.cached_function_calls', verbose=0, mmap_mode='c')

def cache_to_disk(fn):
    '''
    Decorator that enables caching function calls to the disk at 
    `Path(__file__).parent / '..' / '.cached_function_calls'`.
    E.g.

        @cache_to_disk
        def fibonacci(n):
            ...
    '''
    return _cache_instance.cache(fn)


def cache_to_mem(fn):
    '''
    Decorator that acts as an alias of `lru_cache(maxsize=None)`
    E.g.

        @cache_to_mem
        def fibonacci(n):
            ...
    '''
    @lru_cache(maxsize=None)
    def wrapped_fn(*args, **kwargs): 
        return fn(*args, **kwargs)

    return wrapped_fn

# DEBUG: uncomment below line to disable caching by making a dummy wrapper
# cache_to_disk = lambda x: x
