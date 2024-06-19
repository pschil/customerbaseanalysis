"""Classes used for the `foreach_x()` pattern.

Provide functionality to iterate over data elements, apply a method to each of them, and
store the result with identifiers.
"""

import abc
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Any, Callable, Iterator
import modin.pandas as pd


@dataclass
class ForeachDataElement:
    """Element passed from generator (data) to mappings and to fn_df/fn_asis.

    Attributes:
        data: The actual data, such as Decile or PeriodData.
        metainfo: Additional information about the data, such as timestamp until when.
    """

    data: Any
    metainfo: Any


@dataclass
class ForeachIdMapping:
    """How the content of ID cols is generated at each iteration ."""

    fn_id: Callable[[int, ForeachDataElement], Any]
    name: str


@dataclass
class ForeachKeyMapping:
    """How the key is generated at each iteration."""

    fn_key: Callable[[int, ForeachDataElement], Hashable]


class ForeachApplier(abc.ABC):
    """Apply function to each data element, store with identifier and collect results."""

    def __call__(self, data: Iterator[ForeachDataElement], **kwargs) -> pd.DataFrame:
        """Iterate over data, apply a method to it and store result with some id/key."""
        for i, elem_i in enumerate(data):
            self._append(elem_i=elem_i, i=i + 1, **kwargs)

        return self._collect()

    @abc.abstractmethod
    def _append(self, elem_i: ForeachDataElement, i: int, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _collect(self):
        raise NotImplementedError


class ForeachAsIs(ForeachApplier):
    """Apply a function to each data element and store the result "as-is" in a dict.

    Attributes:
        fn_asis: Function to apply to each data element.
        key_mapping: Mapping to generate the dict key for each data element.
        results: Dict to store results.
    """

    results: dict[Hashable, Any]

    def __init__(
        self, fn_asis: Callable[[Any], Any], key_mapping: ForeachKeyMapping
    ) -> None:
        self.fn_asis = fn_asis
        self.key_mapping = key_mapping
        self.results = {}

    def _append(self, elem_i: ForeachDataElement, i: int, **kwargs) -> None:
        key_i = self.key_mapping.fn_key(i, elem_i)
        # Verify key does not exist already
        if key_i in self.results:
            raise ValueError(f"Key {key_i} already exists.")

        self.results[key_i] = self.fn_asis(elem_i.data, **kwargs)

    def _collect(self) -> dict:
        if len(self.results) == 0:
            return {}
        else:
            return self.results


class ForeachDF(ForeachApplier):
    """Apply a function to each data element and store the result as a DataFrame.

    Attributes:
        fn_df: Function to apply to each data element.
        id_mappings: Mappings to generate the id columns for each data element.
        results: DataFrames resulting from each iteration.
    """

    results: list[pd.DataFrame]

    def __init__(
        self,
        fn_df: Callable[[Any], pd.DataFrame | dict[str, Any]],
        id_mappings: list[ForeachIdMapping],
    ):
        self.fn = fn_df
        self.id_mappings = id_mappings
        self.results = []

    def _append(self, elem_i: ForeachDataElement, i: int, **kwargs) -> None:
        """Store result of applying self.fn(data) as DataFrame."""

        # Generate result from applying user-defined function to data element
        result_i = self.fn(elem_i.data, **kwargs)
        if not isinstance(result_i, pd.DataFrame):
            result_i = pd.DataFrame([result_i])

        # Generate id columns from self.id_mappings and add as columns. Because they
        # are always added at the beginning, need to insert them in reverse order.
        for id_m in reversed(self.id_mappings):
            result_i.insert(0, column=id_m.name, value=id_m.fn_id(i, elem_i))

        self.results.append(result_i)

    def _collect(self):
        if len(self.results) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(self.results, ignore_index=True)


def verify_single_df_asis(df: Callable, asis: Callable) -> None:
    """Verify that exactly one of df and asis is not None."""
    if sum([df is not None, asis is not None]) != 1:
        raise ValueError("Exactly one of 'df' and 'asis' must be provided.")
