import glob
import json
import logging
import os
import time
from uuid import uuid4

import openai

from .file_writing import try_make, try_open

LOCAL_CACHE = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def user(content: str) -> dict:
    return dict(role="user", content=content)


def system(content: str) -> dict:
    return dict(role="system", content=content)


def assistant(content: str) -> dict:
    return dict(role="assistant", content=content)


def load_template(filepath: str) -> str:
    with open(filepath, "r") as f:
        description, text = f.read().split("\n", 1)
        if not description.startswith("#"):
            # Not a proper prompt file
            logger.warning(
                f"File {filepath} does not start with a `# description line`."
            )
            text = description + "\n" + text
        return text.strip()

API_PARAMS = dict(
    model="",
    deployment_id=None,
    max_tokens=256,
    temperature=1,
    top_p=1,
    n=1,
    stop=None,
    presence_penalty=0,
    frequency_penalty=0,
    best_of=1,
    logprobs=None,
    logit_bias=None,
)


def set_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logger.setLevel(logging.INFO)


def catch(fn):
    """
    Function annotation that will wait and re-attempt an API call to OpenAI.

    Raises:
        openai.RateLimitError: After 10 rate-limiting errors.
        openai.APIStatusError: If a non-500 HTTP code is returned or 10 failed attempts.
    """
    attempts = 0

    def wrapper(*args, **kwargs):
        nonlocal attempts
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError as e:
            logger.info("Rate limit error")
            attempts += 1
            if attempts > 9:
                raise e
            time.sleep(min(2**attempts, 60))
            return wrapper(*args, **kwargs)
        except openai.APIStatusError as e:
            logger.info("API Error: " + str(e))
            attempts += 1
            if e.code and int(e.code) >= 500 and attempts <= 9:
                time.sleep(min(2**attempts, 60))
                return wrapper(*args, **kwargs)
            else:
                raise e
        finally:
            attempts = 0

    return wrapper


def save_tmp_cache(base, t, c, result):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.{str(uuid4())}.tmp.json")

    if not os.path.exists(folder_name):
        try_make(folder_name)

    with try_open(cache_file, "w") as w:
        json.dump({t: {c: result}}, w)



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


def to_type_name(_type: str):
    if "list" in _type or "tuple" in _type:
        return "array", {"items": {"type": "string"}}

    return {
        "str": "string",
        "int": "number",
    }.get(_type, _type), {}


def format(s: str, **kwargs):
    for key, value in kwargs.items():
        s = s.replace("{" + key + "}", str(value))
    return s
