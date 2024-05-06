"""Classes for cohort-based analysis."""

import copy
import abc
from typing import Iterable, Callable, Any

import pandas as pd
from customerbaseanalysis.data import OrderSummary, CustomerSummary, PeriodData
from customerbaseanalysis.ctime import CohortTime, CalendarTime
from customerbaseanalysis.mixins import (
    AccessCustomerSummaryPropertiesMixin,
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
    "CohortSet",
    "AcquisitionCohortSplitter",
]


class Cohort(AccessCustomerSummaryPropertiesMixin, AccessOrderSummaryPropertiesMixin):
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
            f"Total Margin: {round(self.sum_margin, 2)}\n"
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

    def summary(self) -> pd.DataFrame:
        """Summary statistics of the cohort."""
        df = self.customer_summary.summary()
        df.insert(loc=0, column="cohort", value=self.name)
        df.insert(loc=1, column="first_order", value=self.time_first_order)
        df.insert(loc=2, column="last_order", value=self.time_last_order)
        return df

    def select_customers(self, customer_ids: Iterable) -> "Cohort":
        """Cohort of customers with the given customer_ids."""
        # Verify all customer_ids are in the cohort
        if not set(customer_ids).issubset(self.customer_ids):
            raise ValueError("Not all customer_ids are in the cohort.")

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

    def select_orders_nth(self, lessthan=None, morethan=None, exactly=None) -> "Cohort":
        """Cohort with orders lessthan/morethan/exactly the n-th of each customer"""
        return Cohort(
            name=self.name,
            acq_period=self.acq_period,
            order_summary=self.order_summary.select_nth(
                lessthan=lessthan, morethan=morethan, exactly=exactly
            ),
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

    def foreach_orders(
        self,
        until: CalendarTime | CohortTime,
        # after: CalendarTime | CohortTime | None = None,
        # on: CalendarTime | CohortTime | None = None,
        df: Callable[[OrderSummary, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[OrderSummary, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Subset orders from start to `until` and apply `df` or `asis` on each subset."""

        verify_single_df_asis(df=df, asis=asis)

        # # Verify only one of until, after, on is not None
        # if sum([until is not None, after is not None, on is not None]) != 1:
        #     raise ValueError("Exactly one of until, after, on must be not None.")

        # c_time = until or after or on
        timestamps = until.to_timestamps(start=self.time_first_acquisition)

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


class CohortSet:
    """Set of acquisition cohorts.

    Set and not list to define sub-classes which store multiple cohorts which were
    acquired at the same time but in different channels.

    Attributes:
        cohorts (dict[str, Cohort]): The cohorts in the set.
    """

    cohorts: dict[str, Cohort]

    def __str__(self) -> str:
        return (
            f"CohortSet\n"
            f"\n"
            f"Number of Cohorts: {self.n_cohorts}\n"
            f"Acquisition Period: {self.time_first_acquisition.date()} <> "
            f"{self.time_last_acquisition.date()}\n"
            f"Order Period: {self.time_first_order.date()} <> {self.time_last_order.date()}\n"
            f"\n"
            f"Num Customers: {self.n_customers}\n"
            f"Num Orders: {self.n_orders}\n"
            f"Total Revenue: {self.sum_revenue}\n"
            f"Total Margin: {self.sum_margin}\n"
        )

    def __init__(self, cohorts: dict[str, Cohort]):
        self.cohorts = copy.deepcopy(cohorts)

    def __getitem__(self, item: str):
        """Get a single cohort with the given name."""
        return self.cohorts[str(item)]

    @property
    def n_cohorts(self) -> int:
        """Number of cohorts."""
        return len(self.cohorts)

    @property
    def cohort_names(self) -> set[str]:
        """Names of the cohorts."""
        return set(self.cohorts.keys())

    @property
    def time_first_acquisition(self) -> pd.Timestamp:
        """The timestamp of the first acquisition across all cohorts."""
        return min(c.time_first_acquisition for c in self.cohorts.values())

    @property
    def time_last_acquisition(self) -> pd.Timestamp:
        """The timestamp of the last acquisition across all cohorts."""
        return max(c.time_last_acquisition for c in self.cohorts.values())

    @property
    def time_first_order(self) -> pd.Timestamp:
        """The timestamp of the first order across all cohorts."""
        return min(c.time_first_order for c in self.cohorts.values())

    @property
    def time_last_order(self) -> pd.Timestamp:
        """The timestamp of the last order across all cohorts."""
        return max(c.time_last_order for c in self.cohorts.values())

    @property
    def n_customers(self) -> int:
        """Number of customers across all cohorts."""
        return sum(c.n_customers for c in self.cohorts.values())

    @property
    def n_orders(self) -> int:
        """Number of orders across all cohorts."""
        return sum(c.n_orders for c in self.cohorts.values())

    @property
    def sum_revenue(self) -> float:
        """Total revenue across all cohorts."""
        return sum(c.sum_revenue for c in self.cohorts.values())

    @property
    def sum_margin(self) -> float:
        """Total margin across all cohorts."""
        return sum(c.sum_margin for c in self.cohorts.values())

    @property
    def df_acquisition(self) -> pd.DataFrame:
        """For each customer: Cohort affiliation and date of first order (acquisition)."""
        return self.foreach_cohort(df=lambda c: c.df_acquisition)[
            ["customer_id", "cohort_name", "time_first_order"]
        ]

    def foreach_cohort(
        self,
        df: Callable[[Cohort, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[Cohort, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Apply a function `df` or `asis` to each cohort."""

        verify_single_df_asis(df=df, asis=asis)

        def get_cohorts():
            for c_data in self.cohorts.values():
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
            pd.concat([c.summary() for c in self.cohorts.values()])
            .sort_values("first_order")
            .reset_index(drop=True)
        )


class CohortSplitter(abc.ABC):
    """Split order data into cohorts."""

    @abc.abstractmethod
    def split(self, order_summary: OrderSummary, **kwargs) -> CohortSet:
        """Split the data into cohorts."""
        raise NotImplementedError

    @classmethod
    def plot_customer_growth(cls, order_summary: OrderSummary):
        """Plot the number of customers acquired in each week."""
        # use cohort data structure to iterate over all periods (weeks)
        c_first_orders = Cohort(
            name="placeholder",
            acq_period=pd.Period("NaT"),
            order_summary=order_summary.select_first_orders(),
        )

        df_plot = c_first_orders.foreach_period(
            periods="W-MON",
            df=lambda p: {"n_acquired": p.n_customers},
        )

        return df_plot.plot(
            x="period",
            y="n_acquired",
            xlabel="Period",
            ylabel="Number of Customers Acquired",
            title="Customer Growth Over Time",
            kind="line",
        )


class AcquisitionCohortSplitter(CohortSplitter):
    """Split customer data into cohorts based on acquisition time."""

    def __init__(self, freq: str):
        self.freq = freq

    def split(self, order_summary: OrderSummary) -> CohortSet:
        """Split the data into cohorts based on customers first order."""

        # Using pandas data structures
        # c_sum = CustomerSummary.from_ordersummary(order_summary)
        # first_orders = c_sum.data[["customer_id", "time_first_order"]]
        # acq_periods = pd.period_range(
        #     start=first_orders["time_first_order"].min(),
        #     end=first_orders["time_first_order"].max(),
        #     freq=self.freq,
        # )
        # cohorts = {}
        # for p in acq_periods:
        #     # customers doing their first order in this period
        #     cid_in_period = first_orders.query(
        #         "time_first_order >= @p.start_time and time_first_order < @p.end_time"
        #     )
        # cid_in_period = set(cid_in_period["customer_id"])
        # cohort_name = str(p)
        # cohorts[cohort_name] = Cohort(
        #     name=cohort_name,
        #     acq_period=p,
        #     order_summary=order_summary.select_customers(cid_in_period),
        # )
        # return CohortSet(cohorts=cohorts)

        # using package data structures
        # use cohort data structure to iterate over all periods (weeks)
        c_first_orders = Cohort(
            name="placeholder",
            acq_period=pd.Period("NaT"),
            order_summary=order_summary.select_first_orders(),
        )

        cids_per_period = c_first_orders.foreach_period(
            periods="M",
            df=lambda p: {"customer_ids": p.customer_ids},
        )

        cohorts = {}
        for p, df in cids_per_period.groupby("period"):
            cid_in_period = df["customer_ids"].squeeze()
            cohort_name = str(p)
            cohorts[cohort_name] = Cohort(
                name=cohort_name,
                acq_period=p,
                order_summary=order_summary.select_customers(cid_in_period),
            )

        return CohortSet(cohorts=cohorts)


# class AcquisitionTableCohortSplitter(AcquisitionCohortSplitter):
#     pass

# class AcquisitionChannelCohortSplitter(AcquisitionCohortSplitter):
#     """Split customer data into cohorts based on acquisition time and channel."""
#     def __init__(self, freq: str, channel: )
