import logging, re
from typing import Optional, TypeVar, TypedDict, Callable
from abc import ABC
from itertools import zip_longest
from multiprocessing import Pool
from threading import Thread

from openai import OpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types import Completion


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Message(TypedDict):
    role: str
    content: str


class FunctionT(TypedDict):
    name: str
    description: str
    parameters: dict[str, dict[str, str | dict]]

Prompt = TypeVar("Prompt", str, list[Message])

class Provider(ABC):
    def call():
        pass
    
    def chat():
        pass
    
    def complete():
        pass
    
    def embed():
        pass


def mp_call(args):
    # 1. check true cache for t&c
    c = args[0][1]
    if c in LOCAL_CACHE:
        return LOCAL_CACHE[c]

    # 2. if not in true cache, compute
    fn = args[1]["__function"]
    fns = {"complete": complete, "chat": chat}
    result = fns.get(fn)(
        args[0][0],
        **{k: v for k, v in args[1].items() if not k.startswith("__")},
        nocache=True,
    )
    # 3. add to tmp cache
    t = json.dumps(args[1]["__template"])
    save_tmp_cache(fn, t, c, result)
    return result


def to_slices(template: Prompt, iters, constants):
    for slices in zip_longest(*iters, fillvalue=constants):
        key_values = {k: v for d in slices for k, v in d.items()}
        yield (format_prompt(template, **key_values), str(key_values))


def format_prompt(template: Prompt, **kwargs):
    if isinstance(template, str):
        return format(template, **kwargs)
    else:
        return [
            {
                "role": message["role"],
                "content": format(message["content"], **kwargs),
            }
            for message in template
        ]


def map_reduce(template: Prompt, n=8, **kwargs):
    params = {}
    for key, value in kwargs.items():
        if key in API_PARAMS:
            params[key] = value
    params["__template"] = template

    if isinstance(template, str):
        params["__function"] = "complete"
    else:
        params["__function"] = "chat"

    # Create an list of slices for each template
    iters = [[]]
    constants = {}
    max_len = 0
    for key, value in kwargs.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            iters.append([{key: v} for v in value])
            max_len = max(max_len, len(value))
        else:
            constants[key] = value

    iterator = to_slices(template, iters, constants)

    try:
        with Pool(
            processes=min(n, max_len),
            initializer=load_cache,
            initargs=(params["__function"], json.dumps(params["__template"])),
        ) as pool:
            for res in pool.imap(mp_call, zip_longest(iterator, [], fillvalue=params)):
                yield res
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("EXCEPTION!", e)
    finally:
        collate_caches(params["__function"])


def thread_map_reduce(template: Prompt, n=8, **kwargs):
    params = {}
    for key, value in kwargs.items():
        if key in API_PARAMS:
            params[key] = value

    if isinstance(template, str):
        fn = complete
    else:
        fn = chat

    # Create an list of slices for each template
    iters = [[]]
    constants = {}
    max_len = 0
    for key, value in kwargs.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            iters.append([{key: v} for v in value])
            max_len = max(max_len, len(value))
        else:
            constants[key] = value

    iterator = to_slices(template, iters, constants)

    results = ["" for _ in range(max_len)]

    def thread_call(message, idx):
        try:
            results[idx] = fn(
                message, **{k: v for k, v in params.items()}, nocache=True
            )
        except RateLimitError:
            results[idx] = ""

    num_threads = min(n, max_len)
    threads = []
    for idx, (msg, _) in enumerate(iterator):
        threads.append(Thread(target=thread_call, args=(msg, idx)))
        threads[-1].start()

        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()

    return results

def parse_doc(doc: str):
    if not doc:
        return "", {}, []

    lines = doc.split(":param:")
    fn_description = lines[0].strip()
    properties = {}
    required = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(":")
        name = parts[0].strip()
        _type = parts[1].strip()
        _type = re.sub(r"Optional\[(.*)\]", r"\1", _type)
        description = ":".join(parts[2:]).strip()
        if name.startswith("*"):
            name = name[1:]
            required.append(name)

        type_name, type_items = to_type_name(_type)
        properties[name] = {
            "type": type_name,
            "description": description,
        }
        if type_items:
            properties[name].update(type_items)

    return fn_description, properties, required

def create_function_call(fn: Callable) -> FunctionT:
    description, properties, required = parse_doc(fn.__doc__)
    return {
        "name": fn.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
