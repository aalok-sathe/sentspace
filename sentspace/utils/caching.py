
from joblib import Memory
from pathlib import Path
from functools import lru_cache

_cache_instance = Memory(location=Path(__file__).parent / '..' / '.cached_function_calls', verbose=0, mmap_mode='c')
cache_to_disk = _cache_instance.cache
cache_to_mem = lru_cache(maxsize=None)

# DEBUG: uncomment below line to disable caching by making a dummy wrapper
# cache_to_disk = lambda x: x
