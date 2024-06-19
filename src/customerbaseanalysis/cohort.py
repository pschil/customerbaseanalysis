"""Classes for cohort-based analysis."""

import copy
import abc
from typing import Iterable, Callable, Any

import modin.pandas as pd
from matplotlib import pyplot as plt

from customerbaseanalysis.perioddata import PeriodData
from customerbaseanalysis.decile import DecileList, DecileSplitter
from customerbaseanalysis.data import OrderSummary, CustomerSummary
from customerbaseanalysis.ctime import CohortTime, CalendarTime, CustomerTime
from customerbaseanalysis.mixins import (
    AccessCustomerSummaryMixin,
    AccessOrderSummaryPropertiesMixin,
)
from customerbaseanalysis.foreach import (
    ForeachDF,
    ForeachAsIs,
    ForeachIdMapping,
    ForeachKeyMapping,
    ForeachDataElement,
    verify_single_df_asis,
)

__all__ = [
    "Cohort",
    "CohortList",
    "AcquisitionCohortSplitter",
]


class Cohort(AccessCustomerSummaryMixin, AccessOrderSummaryPropertiesMixin):
    """Customer and order data of a group of customers acquired in the same period.

    Attributes:
        name (str): The descriptive name of the cohort.
        acq_period (pd.Period): The acquisition period of the cohort.
        order_summary (OrderSummary): Order data of the cohort.
        customer_summary (CustomerSummary): Customer data of the cohort.
    """

    def __init__(self, name: str, acq_period: pd.Period, order_summary: OrderSummary):
        self.name = name
        self.order_summary = copy.deepcopy(order_summary)
        self.customer_summary = CustomerSummary.from_ordersummary(self.order_summary)
        self.acq_period = acq_period

    def __str__(self) -> str:
        """Pretty print this cohort."""
        return (
            f"Cohort\n"
            f"\n"
            f"Name: {self.name}\n"
            f"Acquisition Period: {self.acq_period}\n"
            f"First Orders: {self.time_first_acquisition.date()} <> "
            f"{self.time_last_acquisition.date()}\n"
            f"Order Period: {self.time_first_order.date()} <> {self.time_last_order.date()}\n"
            f"\n"
            f"Customers: {self.n_customers}\n"
            f"Orders: {self.n_orders}\n"
            f"Total Revenue: {round(self.sum_revenue,2)}\n"
            f"Total Profit: {round(self.sum_profit, 2)}\n"
        )

    @property
    def df_acquisition(self) -> pd.DataFrame:
        """Acquisition timestamp ('time_first_order') of each customer_id."""
        return self.customer_summary.data[["customer_id", "time_first_order"]].copy()

    @property
    def time_first_acquisition(self) -> pd.Timestamp:
        """The timestamp when the first customer of this cohort was acquired."""
        return self.df_acquisition["time_first_order"].min()

    @property
    def time_last_acquisition(self) -> pd.Timestamp:
        """The timestamp when the last customer of this cohort was acquired."""
        return self.df_acquisition["time_first_order"].max()

    def rename(self, name: str) -> "Cohort":
        """Return a copy of the cohort with a new name."""
        return Cohort(
            name=name, acq_period=self.acq_period, order_summary=self.order_summary
        )

    def summary(self) -> pd.DataFrame:
        """Summary statistics of the cohort."""
        df = self.customer_summary.summary()
        df.insert(loc=0, column="cohort", value=self.name)
        df.insert(loc=1, column="first_order", value=self.time_first_order)
        df.insert(loc=2, column="last_order", value=self.time_last_order)
        return df

    def select_customers(self, customer_ids: Iterable) -> "Cohort":
        """Cohort of customers with the given customer_ids."""

        return Cohort(
            name=self.name,
            acq_period=self.acq_period,
            order_summary=self.order_summary.select_customers(customer_ids),
        )

    def select_customers_norders(
        self, lessthan=None, morethan=None, exactly=None
    ) -> "Cohort":
        """Cohort of customers which have lessthan/exactly/morethan n orders"""
        n_customer_ids = self.customer_summary.customerids_numorders(
            lessthan=lessthan, morethan=morethan, exactly=exactly
        )
        return self.select_customers(n_customer_ids)

    def select_deciles(
        self, splitter: DecileSplitter, key: str | list[str] | int | slice
    ) -> "Cohort":
        """Cohort of customers in decile(s) selected with `key` from DecileList."""
        deciles = self.to_deciles(splitter)
        return self.select_customers(deciles[key].customer_ids)

    def select_orders_nth(self, lessthan=None, morethan=None, exactly=None) -> "Cohort":
        """Cohort with orders lessthan/morethan/exactly the n-th of each customer"""
        return Cohort(
            name=self.name,
            acq_period=self.acq_period,
            order_summary=self.order_summary.select_nth(
                lessthan=lessthan, morethan=morethan, exactly=exactly
            ),
        )

    def select_orders_until(self, ts: str | pd.Timestamp) -> "Cohort":
        """Cohort with orders until a given timestamp."""
        return Cohort(
            name=self.name,
            acq_period=self.acq_period,
            order_summary=self.order_summary.select_until(ts=ts),
        )

    def select_orders_sincealive(self, timedelta: str | pd.Timedelta) -> "Cohort":
        """Cohort with orders within `timedelta` of first order of each customer."""
        return Cohort(
            name=self.name,
            acq_period=self.acq_period,
            order_summary=self.order_summary.select_since_alive(timedelta=timedelta),
        )

    def drop_otb(self) -> "Cohort":
        """Drop all one-time-buyers (customers with only 1 order) from the cohort."""
        return self.select_customers_norders(morethan=1)

    def select_otb(self) -> "Cohort":
        """Select only one-time-buyers (customers with only 1 order) from the cohort."""
        return self.select_customers_norders(exactly=1)

    def select_first_orders(self) -> "Cohort":
        """Select only the first order of each customer."""
        return self.select_orders_nth(exactly=1)

    def select_repeat_orders(self) -> "Cohort":
        """Select only the repeat orders of each customer."""
        return self.select_orders_nth(morethan=1)

    def inter_order_timedelta(self) -> pd.DataFrame:
        """Timedelta to previous order of the same customer"""
        return self.order_summary.apply_inter_order(
            lambda df: df["timestamp"].diff().dropna()
        )

    def to_deciles(self, splitter: DecileSplitter) -> DecileList:
        """Split the cohort into deciles."""
        return splitter.split(self.customer_summary)

    def foreach_period(
        self,
        periods: str | pd.PeriodIndex,
        df: Callable[[PeriodData, Any], pd.DataFrame | dict[str, Any]] | None = None,
        asis: Callable[[PeriodData, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Split order data into periods by `periods` and apply `df` or `asis` on each."""

        verify_single_df_asis(df=df, asis=asis)

        if isinstance(periods, str):
            periods = pd.period_range(
                start=self.time_first_order,
                end=self.time_last_order,
                freq=periods,
            )

        def get_period_data():
            for p in periods:
                period_data = PeriodData.from_ordersummary(
                    name=str(p), period=p, order_summary=self.order_summary
                )
                yield ForeachDataElement(data=period_data, metainfo=p)

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="period_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(
                        name="period_name", fn_id=lambda _, fde: fde.data.name
                    ),
                    ForeachIdMapping(name="period", fn_id=lambda _, fde: fde.metainfo),
                    ForeachIdMapping(
                        name="period_start_time",
                        fn_id=lambda _, fde: fde.metainfo.start_time,
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=lambda _, fde: fde.data.name),
            )

        return foreach(get_period_data(), **kwargs)

    def foreach_rolling_periods(
        self,
        periods: str | list[pd.Period] | pd.PeriodIndex,
        window: int,
        df: Callable[[tuple[PeriodData, ...], Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[tuple[PeriodData, ...], Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Roll a window of periods over orders and apply `df` or `asis` on each window."""
        # Possible alternative names: foreach_rolling_period_window,
        # foreach_rolling_window, foreach_period_window

        verify_single_df_asis(df=df, asis=asis)

        if isinstance(periods, str):
            periods = pd.period_range(
                start=self.time_first_order,
                end=self.time_last_order,
                freq=periods,
            )

        def get_rolling_period_data():
            # alternative: Use a double ended queue with fixed length which discards
            # elements at as they are added
            for i in range(len(periods) - window + 1):
                # periods in the window
                periods_i = periods[i : i + window]

                # Create the data of each period in the window
                period_data_i = [
                    PeriodData.from_ordersummary(
                        name=str(p), period=p, order_summary=self.order_summary
                    )
                    for p in periods_i
                ]
                yield ForeachDataElement(data=period_data_i, metainfo=periods_i)

        def win_name(_, fde):
            return "/".join(p.name for p in fde.data)

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="window_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(name="window_name", fn_id=win_name),
                    ForeachIdMapping(
                        name="window_first_period",
                        fn_id=lambda _, fde: min(fde.metainfo),
                    ),
                    ForeachIdMapping(
                        name="window_last_period",
                        fn_id=lambda _, fde: max(fde.metainfo),
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=win_name),
            )

        return foreach(get_rolling_period_data(), **kwargs)

    def foreach_orders_until(
        self,
        ctime: CalendarTime | CohortTime,
        df: Callable[[OrderSummary, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[OrderSummary, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Subset orders from start to `ctime` and apply `df` or `asis` on each subset."""
        # Further future methods: foreach_orders_until(), foreach_orders_on()

        verify_single_df_asis(df=df, asis=asis)

        timestamps = ctime.to_timestamps(start=self.time_first_acquisition)

        def get_orders_until():
            for ts in timestamps:
                yield ForeachDataElement(
                    data=self.order_summary.select_until(ts), metainfo=ts
                )

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="until_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(
                        name="until_time", fn_id=lambda _, fde: fde.metainfo
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=lambda _, fde: fde.metainfo),
            )

        return foreach(get_orders_until(), **kwargs)

    def foreach_orders_sincealive(
        self,
        deltas: CustomerTime,
        df: Callable[[OrderSummary, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[OrderSummary, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Subset to orders within `deltas` of first order of each customer."""

        verify_single_df_asis(df=df, asis=asis)

        def get_orders_sincealive():
            for td in deltas.deltas:
                yield ForeachDataElement(
                    data=self.order_summary.select_since_alive(timedelta=td),
                    metainfo=td,
                )

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="sincealive_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(
                        name="sincealive_delta", fn_id=lambda _, fde: fde.metainfo
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=lambda _, fde: fde.metainfo),
            )

        return foreach(get_orders_sincealive(), **kwargs)

    # def plot_periods(
    #     self, periods: str, fn: Callable[[PeriodData, Any], float], ax=None, **kwargs
    # ):
    #     """Line plot where the value for each period is produced by `fn(period)`.
    #     (x,y) = (period, fn(period))
    #     """
    #     # pylint: disable=C0415
    #     import seaborn as sns
    #     df_plot = self.foreach_period(
    #         periods=periods,
    #         df=fn,
    #     )
    #     return df_plot
    #     return sns.lineplot(data=df_plot, x="period", y="value",ax=ax, **kwargs)
    #     # return df_plot.plot(x="period", y="value", xlabel="Period", ax=ax, **kwargs)


class CohortList:
    """List of acquisition cohorts.

    Attributes:
        cohorts: The cohorts, ordered by acquisition period.
    """

    cohorts: list[Cohort]

    def __str__(self) -> str:
        return (
            f"CohortList\n"
            f"\n"
            f"Number of Cohorts: {self.n_cohorts}\n"
            f"Acquisition Period: {self.time_first_acquisition.date()} <> "
            f"{self.time_last_acquisition.date()}\n"
            f"Order Period: {self.time_first_order.date()} <> {self.time_last_order.date()}\n"
            f"\n"
            f"Num Customers: {self.n_customers}\n"
            f"Num Orders: {self.n_orders}\n"
            f"Total Revenue: {round(self.sum_revenue, 2)}\n"
            f"Total Profit: {round(self.sum_profit, 2)}\n"
        )

    def __init__(self, cohorts: list[Cohort]):
        # Verify list is not empty
        if not cohorts:
            raise ValueError("List of cohorts must not be empty.")
        # Verify all cohort names are unique
        if len(set(c.name for c in cohorts)) != len(cohorts):
            raise ValueError("All cohort names must be unique.")
        # Verify ordered by acquisition period
        if not all(
            cohorts[i].acq_period.start_time <= cohorts[i + 1].acq_period.start_time
            for i in range(len(cohorts) - 1)
        ):
            raise ValueError("Cohorts must be ordered (<=) by acquisition period.")

        self.cohorts = copy.deepcopy(cohorts)

    def __getitem__(self, item: str | list[str] | int | slice) -> "Cohort | CohortList":
        """Get a cohort or a subset of cohorts by position(s) or name(s)."""
        if isinstance(item, str):
            # find cohort by name
            return next(c for c in self.cohorts if c.name == item)

        elif isinstance(item, int):
            return self.cohorts[item]

        elif isinstance(item, list):
            # Verify all names are in the cohort (not silently ignoring some items)
            if not set(item).issubset(self.cohort_names):
                raise KeyError("Not all cohort names are in this cohort list.")

            # find cohorts by names
            return CohortList([c for c in self.cohorts if c.name in item])

        elif isinstance(item, slice):
            return CohortList(self.cohorts[item])

        else:
            raise TypeError("Invalid type for item.")

    def __iter__(self):
        """Provide Iterator to iterate over the cohorts."""
        # this makes it an "Iterable"
        return iter(self.cohorts)

    @property
    def n_cohorts(self) -> int:
        """Number of cohorts."""
        return len(self.cohorts)

    @property
    def cohort_names(self) -> list[str]:
        """Names of the cohorts."""
        return [c.name for c in self]

    @property
    def acq_periods(self) -> list[pd.Period]:
        """Acquisition periods of the cohorts."""
        return [c.acq_period for c in self]

    @property
    def time_first_acquisition(self) -> pd.Timestamp:
        """The timestamp of the first acquisition across all cohorts."""
        return min(c.time_first_acquisition for c in self)

    @property
    def time_last_acquisition(self) -> pd.Timestamp:
        """The timestamp of the last acquisition across all cohorts."""
        return max(c.time_last_acquisition for c in self)

    @property
    def time_first_order(self) -> pd.Timestamp:
        """The timestamp of the first order across all cohorts."""
        return min(c.time_first_order for c in self)

    @property
    def time_last_order(self) -> pd.Timestamp:
        """The timestamp of the last order across all cohorts."""
        return max(c.time_last_order for c in self)

    @property
    def n_customers(self) -> int:
        """Number of customers across all cohorts."""
        return sum(c.n_customers for c in self)

    @property
    def n_orders(self) -> int:
        """Number of orders across all cohorts."""
        return sum(c.n_orders for c in self)

    @property
    def sum_revenue(self) -> float:
        """Total revenue across all cohorts."""
        return sum(c.sum_revenue for c in self)

    @property
    def sum_profit(self) -> float:
        """Total profit across all cohorts."""
        return sum(c.sum_profit for c in self)

    @property
    def df_acquisition(self) -> pd.DataFrame:
        """For each customer: Cohort affiliation and date of first order (acquisition)."""
        return self.foreach_cohort(df=lambda c: c.df_acquisition)[
            ["customer_id", "cohort_name", "time_first_order"]
        ]

    def select_period_acquired(
        self,
        within: str | pd.Period | None = None,
        lower: str | pd.Period | None = None,
        upper: str | pd.Period | None = None,
        between: tuple[str, str] | tuple[pd.Period, pd.Period] | None = None,
    ) -> "CohortList":
        """Select cohorts based on the acquisition period.

        Args:
            within: Cohorts acquired within this period (inclusive).
            lower: Cohorts acquired on or after this period (inclusive).
            upper: Cohorts acquired on or before this period (inclusive).
            between: Cohorts acquired between these periods (inclusive).
        """
        # Verify only one given
        num_given = sum(
            [
                within is not None,
                lower is not None,
                upper is not None,
                between is not None,
            ]
        )
        if num_given != 1:
            raise ValueError("Exactly one of parameter may be given.")

        if within is not None:
            within = pd.Period(within)
            cohorts = [
                c
                for c in self
                if c.acq_period.start_time >= within.start_time
                and c.acq_period.end_time <= within.end_time
            ]
        elif lower is not None:
            lower = pd.Period(lower)
            cohorts = [c for c in self if c.acq_period >= lower]
        elif upper is not None:
            upper = pd.Period(upper)
            cohorts = [c for c in self if c.acq_period <= upper]
        else:
            between = tuple(pd.Period(p) for p in between)
            # Verify first period is before second period
            if between[0] >= between[1]:
                raise ValueError("First period must be before second period.")
            cohorts = [c for c in self if between[0] <= c.acq_period <= between[1]]

        return CohortList(cohorts=cohorts)

    def select_deciles(
        self, splitter: DecileSplitter, key: str | list[str] | int
    ) -> "CohortList":
        """From each cohort, select customers in the decile(s) selected with `key`."""
        return CohortList([c.select_deciles(splitter=splitter, key=key) for c in self])

    def foreach_cohort(
        self,
        df: Callable[[Cohort, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[Cohort, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Apply a function `df` or `asis` to each cohort."""

        verify_single_df_asis(df=df, asis=asis)

        def get_cohorts():
            for c_data in self:
                yield ForeachDataElement(data=c_data, metainfo=c_data.name)

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="cohort_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(
                        name="cohort_time_acq",
                        fn_id=lambda _, fde: fde.data.time_first_acquisition,
                    ),
                    ForeachIdMapping(
                        name="cohort_name", fn_id=lambda _, fde: fde.metainfo
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=lambda _, fde: fde.metainfo),
            )

        return foreach(get_cohorts(), **kwargs)

    def summary(self) -> pd.DataFrame:
        """Summary statistics across all cohorts."""
        return (
            pd.concat([c.summary() for c in self])
            .sort_values("first_order")
            .reset_index(drop=True)
        )

    def plot_periods_stackedbar(
        self,
        periods: str | pd.PeriodIndex,
        fn: Callable[[PeriodData], float],
        bar_fmt: str | Callable[[float], str] = "%g",
        **kwargs,
    ) -> plt.Axes:
        """Bar plot where the value for each cohort and period is produced by `fn`."""
        df_plot = self.foreach_cohort(df=lambda c: c.foreach_period(periods, df=fn))
        # fn returns float which is saved in column named 0 (int, not str)
        df_plot = df_plot.rename(columns={0: "metric"})
        df_plot = df_plot.pivot(
            index="period",
            columns="cohort_name",
            values="metric",
        ).sort_index()

        ax = df_plot.plot.bar(stacked=True, **kwargs)

        # # Add totals on top (last container)
        # total = df_plot.sum(axis=1)
        # sum values in containers in case ordering is different from df_plot
        totals = pd.DataFrame([barc.datavalues for barc in ax.containers]).sum()
        ax.bar_label(ax.containers[-1], labels=totals, label_type="edge")

        # Add value of each bar patch
        for cont_i in ax.containers:
            ax.bar_label(cont_i, fmt=bar_fmt, label_type="center")

        return ax


class CohortSplitter(abc.ABC):
    """Split order data into cohorts."""

    @abc.abstractmethod
    def split(self, order_summary: OrderSummary, **kwargs) -> CohortList:
        """Split the data into cohorts."""
        raise NotImplementedError

    @classmethod
    def plot_customer_growth(
        cls,
        order_summary: OrderSummary,
        periods: str | pd.PeriodIndex = "M",
        ax: plt.Axes | None = None,
        **kwargs,
    ):
        """Plot the number of customers acquired in each week."""
        # use cohort data structure to iterate over all periods (weeks)
        c_first_orders = Cohort(
            name="placeholder",
            acq_period=pd.Period("NaT"),
            order_summary=order_summary.select_first_orders(),
        )

        df_plot = c_first_orders.foreach_period(
            periods=periods,
            df=lambda p: {"n_acquired": p.n_customers},
        )

        return df_plot.plot(
            x="period",
            y="n_acquired",
            xlabel="Period",
            ylabel="Number of Customers Acquired",
            title="Customer Growth Over Time",
            kind="line",
            ax=ax,
            **kwargs,
        )


class AcquisitionCohortSplitter(CohortSplitter):
    """Split customer data into cohorts based on acquisition time.

    Attributes:
        freq (str): Frequency of acquisition periods.
        min_size (int | None): Minimum number of customers required for a cohort.
    """

    def __init__(self, freq: str, min_size: int | None = None):
        self.freq = freq
        self.min_size = min_size

    def split(self, order_summary: OrderSummary) -> CohortList:
        """Split the data into cohorts based on customers first order."""

        # Using pandas data structures
        c_sum = CustomerSummary.from_ordersummary(order_summary)
        df_first_orders = c_sum.data[["customer_id", "time_first_order"]]
        acq_periods = pd.period_range(
            start=df_first_orders["time_first_order"].min(),
            end=df_first_orders["time_first_order"].max(),
            freq=self.freq,
        )
        cohorts = []

        for p in acq_periods:
            # customers doing their first order in this period
            # has to be inclusive boundaries [period.start_time, period.end_time],
            # because periods are none-overlapping and otherwise customers with their
            # first order on p.end_time are lost
            df_in_period = df_first_orders.query(
                "time_first_order >= @p.start_time and time_first_order <= @p.end_time"
            )

            # Skip if nobody came alive in this period
            if df_in_period.shape[0] == 0:
                continue

            # Skip if minimum num customers required and not fulfilled
            if self.min_size and df_in_period.shape[0] < self.min_size:
                continue

            cids_in_period = set(df_in_period["customer_id"])

            cohorts.append(
                Cohort(
                    name=str(p),
                    acq_period=p,
                    order_summary=order_summary.select_customers(cids_in_period),
                )
            )

        return CohortList(cohorts=cohorts)

        # # using only package data structures
        # # use cohort data structure to iterate over periods
        # c_first_orders = Cohort(
        #     name="placeholder",
        #     acq_period=pd.Period("NaT"),
        #     order_summary=order_summary.select_first_orders(),
        # )

        # cids_per_period = c_first_orders.foreach_period(
        #     periods=self.freq,
        #     df=lambda p: {"customer_ids": p.customer_ids},
        # )

        # cohorts = []
        # for p, df in cids_per_period.groupby("period"):
        #     df_in_period = df["customer_ids"].squeeze()

        #     # skip if too small
        #     if self.min_size is not None and len(cids_per_period) < self.min_size:
        #         continue

        #     cohort_name = str(p)
        #     cohorts.append(
        #         Cohort(
        #             name=cohort_name,
        #             acq_period=p,
        #             order_summary=order_summary.select_customers(df_in_period),
        #         )
        #     )

        # return CohortList(cohorts=cohorts)


# class AcquisitionTableCohortSplitter(AcquisitionCohortSplitter):
#     pass

# class AcquisitionChannelCohortSplitter(AcquisitionCohortSplitter):
#     """Split customer data into cohorts based on acquisition time and channel."""
#     def __init__(self, freq: str, channel: )
