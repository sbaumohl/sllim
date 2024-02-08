import os, logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def try_make(folder_name: str):
    """Attempt to create folder, and log if there is a failure to do so.

    Args:
        folder_name (str): folder name
    """
    try:
        os.makedirs(folder_name, exist_ok=True)
    except Exception:
        logger.info(f"Cannot create folder {folder_name}")


class fake_file:
    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        return "{}"


@contextmanager
def try_open(filename: str, mode="r"):
    """Attempts to read a file, returning a prop file should the file fail to open.
    After opening, it yields file to the caller, and closes the file when finished. 

    Args:
        filename str: file to attempt to open
        mode (str, optional): Mode to open the file with. 
        See documentation for python `open` for more details. Defaults to "r".

    Yields:
        (IO[Any], fake_file): The file that was opened. If file failed to open, fake_file was yielded.
    """
    try:
        f = open(filename, mode, encoding="utf-8")
    except Exception:
        f = fake_file()

    try:
        yield f

    finally:
        f.close()