from typing import Callable, TypeVar, Iterable, Iterator, Optional, ParamSpec, Generic
from functools import wraps, reduce
import operator
import torch
import numpy as np


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
P = ParamSpec('P')
R = TypeVar('R')


def compose(*functions: Callable) -> Callable:
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(value: A, *functions: Callable[[A], A]) -> A:
    return reduce(lambda v, f: f(v), functions, value)


def curry(fn: Callable) -> Callable:
    @wraps(fn)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= fn.__code__.co_argcount:
            return fn(*args, **kwargs)
        return lambda *a, **k: curried(*(args + a), **dict(kwargs, **k))
    return curried


def memoize(fn: Callable[P, R]) -> Callable[P, R]:
    cache: dict = {}
    
    @wraps(fn)
    def memoized(*args: P.args, **kwargs: P.kwargs) -> R:
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    
    return memoized


def partial_right(fn: Callable, *args) -> Callable:
    @wraps(fn)
    def wrapper(*a):
        return fn(*a, *args)
    return wrapper


def juxt(*functions: Callable[[A], B]) -> Callable[[A], tuple[B, ...]]:
    return lambda x: tuple(f(x) for f in functions)


def complement(predicate: Callable[[A], bool]) -> Callable[[A], bool]:
    @wraps(predicate)
    def complemented(x: A) -> bool:
        return not predicate(x)
    return complemented


def identity(x: A) -> A:
    return x


def const(value: A) -> Callable[..., A]:
    return lambda *_: value


def take(n: int, iterable: Iterable[A]) -> Iterator[A]:
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


def drop(n: int, iterable: Iterable[A]) -> Iterator[A]:
    iterator = iter(iterable)
    for _ in range(n):
        next(iterator, None)
    yield from iterator


def partition(predicate: Callable[[A], bool], iterable: Iterable[A]) -> tuple[list[A], list[A]]:
    true_items, false_items = [], []
    for item in iterable:
        (true_items if predicate(item) else false_items).append(item)
    return true_items, false_items


def chunk(n: int, iterable: Iterable[A]) -> Iterator[list[A]]:
    chunk_items = []
    for item in iterable:
        chunk_items.append(item)
        if len(chunk_items) == n:
            yield chunk_items
            chunk_items = []
    if chunk_items:
        yield chunk_items


def sliding_window(n: int, iterable: Iterable[A]) -> Iterator[tuple[A, ...]]:
    iterator = iter(iterable)
    window = list(take(n, iterator))
    if len(window) == n:
        yield tuple(window)
    for item in iterator:
        window = window[1:] + [item]
        yield tuple(window)


def repeatedly(fn: Callable[[], A], n: Optional[int] = None) -> Iterator[A]:
    if n is None:
        while True:
            yield fn()
    else:
        for _ in range(n):
            yield fn()


def iterate(fn: Callable[[A], A], initial: A) -> Iterator[A]:
    value = initial
    while True:
        yield value
        value = fn(value)


class Maybe(Generic[A]):
    
    def __init__(self, value: Optional[A]):
        self._value = value
    
    @staticmethod
    def of(value: A) -> 'Maybe[A]':
        return Maybe(value)
    
    @staticmethod
    def empty() -> 'Maybe':
        return Maybe(None)
    
    def map(self, fn: Callable[[A], B]) -> 'Maybe[B]':
        return Maybe(fn(self._value) if self._value is not None else None)
    
    def flat_map(self, fn: Callable[[A], 'Maybe[B]']) -> 'Maybe[B]':
        return fn(self._value) if self._value is not None else Maybe.empty()
    
    def filter(self, predicate: Callable[[A], bool]) -> 'Maybe[A]':
        return self if self._value is not None and predicate(self._value) else Maybe.empty()
    
    def get_or_else(self, default: A) -> A:
        return self._value if self._value is not None else default
    
    def is_present(self) -> bool:
        return self._value is not None


class Either(Generic[A]):
    
    def __init__(self, value: A | Exception, is_right: bool = True):
        self._value = value
        self._is_right = is_right
    
    @staticmethod
    def right(value: A) -> 'Either[A]':
        return Either(value, True)
    
    @staticmethod
    def left(error: Exception) -> 'Either':
        return Either(error, False)
    
    def map(self, fn: Callable[[A], B]) -> 'Either[B]':
        if self._is_right:
            try:
                return Either.right(fn(self._value))
            except Exception as e:
                return Either.left(e)
        return self
    
    def flat_map(self, fn: Callable[[A], 'Either[B]']) -> 'Either[B]':
        return fn(self._value) if self._is_right else self
    
    def get_or_else(self, default: A) -> A:
        return self._value if self._is_right else default
    
    def is_right(self) -> bool:
        return self._is_right


def safe(fn: Callable[P, R]) -> Callable[P, Either[R]]:
    @wraps(fn)
    def safe_fn(*args: P.args, **kwargs: P.kwargs) -> Either[R]:
        try:
            return Either.right(fn(*args, **kwargs))
        except Exception as e:
            return Either.left(e)
    return safe_fn


def tensor_map(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    @wraps(fn)
    def mapped(x: torch.Tensor) -> torch.Tensor:
        return fn(x)
    return mapped


def array_map(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    @wraps(fn)
    def mapped(x: np.ndarray) -> np.ndarray:
        return fn(x)
    return mapped


def parallel_map(fn: Callable[[A], B], iterable: Iterable[A], n_workers: int = 4) -> list[B]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(fn, iterable))


def timed(fn: Callable[P, R]) -> Callable[P, tuple[R, float]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        import time
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper