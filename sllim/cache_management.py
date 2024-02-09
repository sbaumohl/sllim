import os
import json
import glob
from .file_writing import try_make, try_open

folder_name = ".cache"


def cache(fn):
    """
    Function annotation that caches a function call and its response, saving to a json file.
    Keys are constructed from all function arguments.
    """
    cache_file = os.path.join(folder_name, f"{fn.__name__}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    def wrapper(*args, **kwargs):
        if kwargs.get("nocache", False):
            kwargs.pop("nocache", None)
            return fn(*args, **kwargs)

        key = str(args + tuple(kwargs.items()))
        if key not in cache:
            res = fn(*args, **kwargs)
            cache[key] = res
            with try_open(cache_file, "w") as w:
                json.dump(cache, w)
        return cache[key]

    return wrapper


def check_cache(base, t, c):
    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    if t in cache and c in cache[t]:
        return cache[t][c]

    return None


def load_cache(base, t):
    global LOCAL_CACHE

    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    LOCAL_CACHE = cache.get(t, {})


def collate_caches(function_name):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{function_name}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    with try_open(cache_file) as w:
        cache = json.load(w)

    for f in glob.glob(os.path.join(folder_name, f"{function_name}.*.tmp.json")):
        with try_open(f) as w:
            tmp_cache = json.load(w)
            key = list(tmp_cache.keys())[0]
            if key not in cache:
                cache[key] = {}
            cache[key].update(tmp_cache[key])
        os.remove(f)

    with try_open(cache_file, "w") as w:
        json.dump(cache, w)
