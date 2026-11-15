"""Intersection and union of several iterables."""

from typing import TypeVar, Iterable, Hashable

T = TypeVar('T', bound=Hashable)


def nested_intersection(iterable_of_iterables: Iterable[Iterable[T]]) -> set[T]:
    """Intersection of all iterables. An empty input gives an empty set. A generator input works."""
    items = list(iterable_of_iterables)
    if not items:
        return set()
    return set(items[0]).intersection(*items[1:])


def nested_union(iterable_of_iterables: Iterable[Iterable[T]]) -> set[T]:
    """Union of all iterables. An empty input gives an empty set. A generator input works."""
    return set().union(*iterable_of_iterables)
