import re
from collections import defaultdict

_SPLIT_PATTERNS = [
    (r"^(.*?)(_)(.*)$", '_'),            # Underscore
    (r"^(.*?)(-)(.*)$", '-'),            # Hyphen
    (r"^(.+ )( )(.*)$", ' '),            # Space
    (r"^(.*?)([A-Z].*)$", '')            # CamelCase (separator is empty)
]


def _split_items(items: list[str], include_sep_in_prefix: bool) -> list[tuple[str, str]]:
    """
    Splits each item into (prefix, suffix) pairs based on the provided patterns.
    """
    pairs = []
    for item in items:
        for pattern, sep in _SPLIT_PATTERNS:
            match = re.match(pattern, item)
            if match:
                groups = match.groups()
                if sep:
                    prefix, separator, suffix = groups
                else:
                    prefix, suffix = groups
                    separator = ''
                if prefix and suffix:
                    if include_sep_in_prefix:
                        prefix += separator
                    else:
                        suffix = separator + suffix
                    pairs.append((prefix, suffix))
                break
    return pairs


def detect_suffix_pairs(items: list[str]) -> dict[str, tuple[str, str]]:
    """
    Detects keys for which there are two different suffixes, with a common prefix.
    """
    pairs = _split_items(items, include_sep_in_prefix=False)
    suffix_dict = defaultdict(set)
    for prefix, suffix in pairs:
        suffix_dict[prefix].add(suffix)
    result = {
        key: tuple(sorted(suffixes))
        for key, suffixes in suffix_dict.items()
        if len(suffixes) == 2
    }
    return result


def detect_prefix_pairs(items: list[str]) -> dict[str, tuple[str, str]]:
    """
    Detects keys for which there are two different prefixes, with a common suffix.
    """
    pairs = _split_items(items, include_sep_in_prefix=True)
    suffix_to_prefixes = defaultdict(set)
    for prefix, suffix in pairs:
        suffix_to_prefixes[suffix].add(prefix)
    result = {
        key: tuple(sorted(prefixes))
        for key, prefixes in suffix_to_prefixes.items()
        if len(prefixes) == 2
    }
    return result


if __name__ == '__main__':
    input_list = [
        'bing_from', 'bing_to',  # ('bing', ('_from', '_to'))
        'bing-from', 'bing-to',  # ('bing', ('-from', '-to'))
        'bing from', 'bing to',  # ('bing', (' from', ' to'))
        'bingFrom', 'bingTo',    # ('bing', ('From', 'To'))

        'from_bong', 'to_bong',  # ('bong', ('from_', 'to_'))
        'from-bong', 'to-bong',  # ('bong', ('from-', 'to-'))
        'from bong', 'to bong',  # ('bong', ('from ', 'to '))
        'fromBong', 'toBong',    # ('bong', ('From', 'To'))

        'Aking', 'kingB',
        'kong', 'kong'
    ]

    suffix_pairs = detect_suffix_pairs(input_list)
    prefix_pairs = detect_prefix_pairs(input_list)

    print("Suffix pairs:", suffix_pairs)
    print("Prefix pairs:", prefix_pairs)
