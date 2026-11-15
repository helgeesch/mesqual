"""Names and tuples of pairs, for example of borders."""


class PairCombiner:
    """Builds names and tuples of pairs: as given, reversed or sorted, for example `'C1 - C2'` for a border from C1 to C2.

    Args:
        separator (str): Text between the two parts of a pair name.
    """

    def __init__(self, separator=' - '):
        self.separator = separator

    def get_combo_as_str(self, a: str, b: str) -> str:
        """Return `a` and `b` joined by the separator, for example `'C1 - C2'`."""
        return self.separator.join([a, b])

    def get_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        """Return `(a, b)`."""
        return a, b

    def get_opposite_combo_as_str(self, a: str, b: str) -> str:
        """Return `b` and `a` joined by the separator, for example `'C2 - C1'` for `a='C1'`."""
        return self.separator.join([b, a])

    def get_opposite_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        """Return `(b, a)`."""
        return b, a

    def get_sorted_combo_as_str(self, a: str, b: str) -> str:
        """Return `a` and `b` in sorted order joined by the separator, so both orders give the same name."""
        return self.separator.join(sorted([b, a]))

    def get_sorted_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        """Return `a` and `b` as a tuple in sorted order."""
        return tuple(sorted([b, a]))
