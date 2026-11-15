"""Exceptions of the flag layer."""


class UnknownFlagError(KeyError, ValueError):
    """Raised when no flag spec declares a flag.

    A subclass of both `KeyError` and `ValueError`, so code that catches either keeps working.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ''


class AmbiguousOverrideError(ValueError):
    """Raised when more than one interpreter, override or spec claims the same flag or interpreter.

    Two overrides of one interpreter on one class or on sibling classes,
    two answering interpreters with one spec name, or two specs that give
    one flag name raise this error.
    """


class FlagUnavailableError(ValueError):
    """A flag is not available on a dataset.

    `fetch` raises it for an accepted flag whose availability is `False`. It
    is a subclass of `ValueError`. `Dimension.check_value` raises it when the
    values function of a flag family fetches an unavailable flag. For the
    values of such a family, see `Dimension.values_for`.

    Args:
        flag: The flag.
        dataset: The name of the dataset.
        reason: Why the flag is not available, as in `Dataset.availability(flag).reason`.

    Attributes:
        flag: The flag.
        dataset: The name of the dataset.
        reason: Why the flag is not available.
    """

    def __init__(self, flag: str, dataset: str, reason: str):
        super().__init__(f'Flag {flag} is not available on dataset {dataset}: {reason}')
        self.flag = flag
        self.dataset = dataset
        self.reason = reason

    def __reduce__(self):
        return type(self), (self.flag, self.dataset, self.reason)
