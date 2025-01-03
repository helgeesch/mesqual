from typing import TypeVar, Iterable, Hashable, Set

T = TypeVar('T', bound=Hashable)


def nested_intersection(iterable_of_iterables: Iterable[Iterable[T]]) -> set[T]:
    size = sum(1 for _ in iterable_of_iterables)
    if size == 0:
        return set()
    my_intersection = set(list(iterable_of_iterables)[0])
    if size == 1:
        return my_intersection
    for i in list(iterable_of_iterables)[1:]:
        my_intersection = my_intersection.intersection(i)
    return my_intersection


def nested_union(iterable_of_iterables: Iterable[Iterable[T]]) -> set[T]:
    size = sum(1 for _ in iterable_of_iterables)
    if size == 0:
        return set()
    my_union = set(list(iterable_of_iterables)[0])
    if size == 1:
        return my_union
    for i in list(iterable_of_iterables)[1:]:
        my_union = my_union.union(i)
    return my_union
