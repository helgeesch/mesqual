"""Base keys that appear with each of several tags."""


class CommonBaseKeyFinder:
    """
    Find the base keys that appear with each of several association tags.

    The base key of a key is the key without the tag. A tag can be anywhere
    in the key, not only at the end.

    Examples:
        >>> finder = CommonBaseKeyFinder('_from', '_to')
        >>> keys = ['node_from', 'node_to', 'node_via', 'country_from', 'country_to', 'capacity']
        >>> sorted(finder.get_keys_for_which_all_association_tags_appear(keys))
        ['country', 'node']
    """

    def __init__(self, *association_tags: str):
        if not association_tags or len(association_tags) < 2:
            raise ValueError('Identifiers must be a list with at least two elements.')
        self._association_tags = association_tags

    def get_keys_for_which_all_association_tags_appear(self, keys: list[str]) -> list[str]:
        """Return the base keys that appear with each tag, in no fixed order.

        The base key of a key is the key with all occurrences of a tag
        removed. The tag can be anywhere in the key, so `'x_from_y'` gives
        the base key `'x_y'` for the tag `'_from'`.
        """
        base_key_sets = [set() for _ in self._association_tags]

        for name in keys:
            for i, association_tag in enumerate(self._association_tags):
                if association_tag in name:
                    base_key_sets[i].add(name.replace(association_tag, ''))

        common_base_keys = set.intersection(*base_key_sets)
        return list(common_base_keys)
