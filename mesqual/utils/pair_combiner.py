class PairCombiner:
    def __init__(self, separator=' - '):
        self.separator = separator

    def get_combo_as_str(self, a: str, b: str) -> str:
        return self.separator.join([a, b])

    def get_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        return a, b

    def get_opposite_combo_as_str(self, a: str, b: str) -> str:
        return self.separator.join([b, a])

    def get_opposite_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        return b, a

    def get_sorted_combo_as_str(self, a: str, b: str) -> str:
        return self.separator.join(sorted([b, a]))

    def get_sorted_combo_as_tuple(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted([b, a]))
