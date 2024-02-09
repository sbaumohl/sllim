import json
import logging
import os
from uuid import uuid4

from .file_writing import try_make, try_open

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


def set_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logger.setLevel(logging.INFO)


def save_tmp_cache(base, t, c, result):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.{str(uuid4())}.tmp.json")

    if not os.path.exists(folder_name):
        try_make(folder_name)

    with try_open(cache_file, "w") as w:
        json.dump({t: {c: result}}, w)


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
