from abc import ABC, abstractmethod
from typing import Any


class BaseMapping(ABC):
    @abstractmethod
    def __call__(self, value) -> Any:
        pass
