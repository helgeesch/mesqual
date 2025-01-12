from typing import Iterable
from abc import ABC, abstractmethod

from mescal.data_sets import DataSet
from mescal.typevars import Flagtype
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    pass


class Validation(ABC):
    @abstractmethod
    def validate(self, data_set: DataSet) -> bool:
        pass

    def get_error_message(self, data_set: DataSet) -> str:
        return f"Validation {self.__class__.__name__} failed for DataSet {data_set.name}."


class DataSetValidator(ABC):
    def __init__(self):
        self.validations: list[Validation] = []
        self._register_validations()

    @abstractmethod
    def _register_validations(self):
        pass

    def add_validations(self, validations: Iterable[Validation]):
        for v in validations:
            self.validations.append(v)

    def add_validation(self, validation: Validation):
        self.validations.append(validation)

    def validate_data_set(self, data_set: DataSet):
        for validation in self.validations:
            if not validation.validate(data_set):
                raise ValidationError(validation.get_error_message(data_set))

        logger.info(f"Success! All validations passed for {self.__class__.__name__} on DataSet {data_set.name}")


class ConstraintValidation(Validation):
    def __init__(
            self,
            flag: Flagtype,
            min_value: float | None = None,
            max_value: float | None = None,
            exact_value: float | None = None,
            object_subset: list[int | str] | None = None
    ):
        self.flag = flag
        self.min_value = min_value
        self.max_value = max_value
        self.exact_value = exact_value
        self.object_subset = object_subset

    def validate(self, data_set: DataSet) -> bool:
        data = data_set.fetch(self.flag)

        if self.object_subset:
            data = data[self.object_subset]

        if self.exact_value is not None:
            return data.eq(self.exact_value).all().all()

        if self.min_value is not None and (data < self.min_value).any().any():
            return False

        if self.max_value is not None and (data > self.max_value).any().any():
            return False

        return True

    def get_error_message(self, data_set: DataSet) -> str:
        conditions = []
        if self.exact_value is not None:
            conditions.append(f"exactly {self.exact_value}")
        if self.min_value is not None:
            conditions.append(f">= {self.min_value}")
        if self.max_value is not None:
            conditions.append(f"<= {self.max_value}")

        subset_info = f" for objects {self.object_subset}" if self.object_subset else ""
        conditions_text = ' and '.join(conditions)
        message = f"{self.__class__.__name__} failed for DataSet {data_set.name}: \n"
        message += f"{self.flag}{subset_info} must be {conditions_text}"
        return message
