"""Datetime operations in calendar- and cohort-time."""

import abc
import modin.pandas as pd

__all__ = ["CalendarTime", "CohortTime", "CustomerTime"]


class CTime(abc.ABC):
    """Abstract base class for calendar- and cohort-time."""

    @abc.abstractmethod
    def to_timestamps(self, start: str | pd.Timestamp) -> list[pd.Timestamp]:
        """Generate timestamps based on `start`."""
        raise NotImplementedError


class CalendarTime(CTime):
    """Generate timestamps based on calendar-time."""

    def __init__(
        self, timestamps: list[str] | list[pd.Timestamp] | pd.DatetimeIndex | None
    ) -> None:
        # Verify not single string
        if isinstance(timestamps, str):
            raise ValueError("Single string not allowed.")

        self.timestamps = list(pd.to_datetime(timestamps))

    def to_timestamps(self, start: str | pd.Timestamp) -> list[pd.Timestamp]:
        if self.timestamps is not None:
            return self.timestamps


class CustomerTime(CTime):
    """Timedeltas since customers came alive."""

    deltas: list[pd.Timedelta]

    def __init__(self, deltas: list[pd.Timedelta]) -> None:
        self.deltas = deltas

    @classmethod
    def from_deltas(cls, deltas: list[str] | list[pd.Timedelta]) -> "CustomerTime":
        """Create CustomerTime object from time deltas."""
        if isinstance(deltas, str):
            raise ValueError("Single string not allowed for delta.")

        return cls(deltas=[pd.to_timedelta(d) for d in deltas])

    @classmethod
    def from_freq(cls, freq: str | pd.Timedelta, n: int | list[int]) -> "CustomerTime":
        """Create CustomerTime object from frequency and n."""
        # build deltas from freq and n
        if isinstance(n, int):
            n = list(range(1, n + 1))

        freq = pd.to_timedelta(freq)

        return cls(deltas=[freq * n_i for n_i in n])

    def to_timestamps(self, start: str | pd.Timestamp) -> list[pd.Timestamp]:
        """Not really useful anywhere."""
        raise NotImplementedError


class CohortTime(CTime):
    """Generate timestamps based on start of cohort.

    Attributes:
        freq_delta: Period spacing.
        ns: List of
    """

    freq_delta: pd.Timedelta
    ns: list[int] | None
    max: int | pd.Timestamp | None

    def __init__(
        self,
        period_freq: str | pd.Timedelta,
        n: int | list[int] | None = None,
        end: int | str | pd.Timestamp | None = None,
    ) -> None:
        # Either n or max must be provided (freq alone is not enough)
        if n is None and end is None:
            raise ValueError("Either 'n' or 'max' must be provided.")

        self.freq_delta = pd.to_timedelta(period_freq)
        if self.freq_delta <= pd.Timedelta(0):
            raise ValueError("Frequency must be positive.")

        if isinstance(n, list):
            if any([n_i <= 0 for n_i in n]):
                raise ValueError("All elements in n must be positive.")
            self.ns = n
        elif isinstance(n, int):
            if n <= 0:
                raise ValueError("N must be positive.")
            self.ns = list(range(1, n + 1))
        elif n is None:
            self.ns = None
        else:
            raise ValueError("Invalid type for 'n'. Only int or list allowed.")

        self.max = pd.to_datetime(end) if isinstance(end, str) else end

    def to_timestamps(self, start: str | pd.Timestamp) -> list[pd.Timestamp]:
        start = pd.to_datetime(start)

        if self.max is None:
            max_ts = None
        elif isinstance(self.max, pd.Timestamp):
            max_ts = self.max
        elif isinstance(self.max, int):
            max_ts = start + self.max * self.freq_delta
        else:
            raise ValueError(
                "Invalid type for 'max'. Only int or pd.Timestamp allowed."
            )

        if self.ns is None:
            if max_ts is not None:
                # Generate timestamps based on max alone (until max is reached)
                return list(
                    pd.date_range(
                        start=start + self.freq_delta, end=max_ts, freq=self.freq_delta
                    )
                )

            raise ValueError("Either 'n' or 'max' must be provided.")
        else:
            # Generate exactly as specified in self.ns
            timestamps = [start + n * self.freq_delta for n in self.ns]
            # Remove timestamps beyond max_ts, if provided
            if max_ts is not None:
                timestamps = [ts for ts in timestamps if ts <= max_ts]
            return timestamps
