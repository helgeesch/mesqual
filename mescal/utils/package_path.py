import os


def get_abs_source_root_path() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PACKAGE_SOURCE_ROOT_PATH = get_abs_source_root_path()
