from abc import ABC, abstractmethod


class DataBase(ABC):
    @abstractmethod
    def get(self, key, **kwargs):
        pass
    
    @abstractmethod
    def set(self, key, value, **kwargs):
        pass
    
    @abstractmethod
    def key_is_up_to_date(self, key, timestamp=None, **kwargs):
        pass
