class CommonBaseKeyFinder:
    """
    Can identify and extract common base keys from a list of string keys,
    where each base key has multiple associated suffixes (or association_tags).
    This is useful for finding items that share a common base but differ by suffix,
    such as matching groups in data structures.

    Example Usage:
        common_base_key_finder = CommonBaseKeyFinder(['_from', '_to'])
        keys = [
            'node_from', 'node_to', 'node_via',
            'king_from', 'king_to', 'king_dong',
            'kong_from', 'kong_to',
            'bing', 'bong',
        ]
        result = common_base_key_finder.get_keys_for_which_all_association_tags_appear(keys)
        print(result)  # Outputs: ['node', 'king', 'kong']
    """
    def __init__(self, *association_tags: str):
        if not association_tags or len(association_tags) < 2:
            raise ValueError("Identifiers must be a list with at least two elements.")
        self._association_tags = association_tags

    def get_keys_for_which_all_association_tags_appear(self, keys: list[str]) -> list[str]:
        base_key_sets = [set() for _ in self._association_tags]

        for name in keys:
            for i, association_tag in enumerate(self._association_tags):
                if association_tag in name:
                    base_key_sets[i].add(name.replace(association_tag, ''))

        common_base_keys = set.intersection(*base_key_sets)
        return list(common_base_keys)


if __name__ == '__main__':
    sample_keys = [
        'node_from', 'node_to', 'node_via',
        'king_from', 'king_to', 'king_dong',
        'kong_from', 'kong_to',
        'bing', 'bong',
    ]

    common_base_key_finder = CommonBaseKeyFinder('_from', '_to', '_via')
    result = common_base_key_finder.get_keys_for_which_all_association_tags_appear(sample_keys)
    print(result)  # Outputs: ['node']

    common_base_key_finder = CommonBaseKeyFinder('_from', '_to')
    result = common_base_key_finder.get_keys_for_which_all_association_tags_appear(sample_keys)
    print(result)  # Outputs: ['node', 'king', 'kong']
