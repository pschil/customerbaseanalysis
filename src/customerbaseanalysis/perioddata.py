"""Data of a single period."""

import modin.pandas as pd

from customerbaseanalysis.data import CustomerSummary, OrderSummary
from customerbaseanalysis.decile import DecileSplitter, DecileList
from customerbaseanalysis.mixins import (
    AccessCustomerSummaryMixin,
    AccessOrderSummaryPropertiesMixin,
)

__all__ = ["PeriodData"]


class PeriodData(AccessCustomerSummaryMixin, AccessOrderSummaryPropertiesMixin):
    """Data of a single period.

    Keeps together order and customer data of a single period.

    Attributes:
        name (str): Descriptive name for this period.
        period (pd.Period): Period this data belongs to.
        order_summary (OrderSummary): Orders in this period.
        customer_summary (CustomerSummary): CustomerSummary of the order data in this
            period (generated from order_summary)
    """

    def __str__(self) -> str:
        """Pretty print data in this period."""
        return (
            f"Period\n"
            f"\n"
            f"Name: {self.name}\n"
            f"Time Period: {self.period}\n"
            f"Customers: {self.n_customers}\n"
            f"Orders ({self.n_orders}): {self.time_first_order.date()} <> "
            f"{self.time_last_order.date()}\n"
            f"Total Revenue: {round(self.sum_revenue, 2)}\n"
            f"Total Profit: {round(self.sum_profit, 2)}\n"
        )

    def __init__(
        self,
        name: str,
        period: pd.Period,
        order_summary: OrderSummary,
        customer_summary: CustomerSummary,
    ):
        self.name = name
        self.period = period
        self.order_summary = order_summary
        self.customer_summary = customer_summary

    @classmethod
    def from_ordersummary(
        cls, name: str, period: pd.Period | str, order_summary: OrderSummary
    ) -> "PeriodData":
        """PeriodData from the order data that falls into the given period."""

        order_summary = order_summary.select_period(period)
        customer_summary = CustomerSummary.from_ordersummary(order_summary)
        return cls(
            name=name,
            period=period,
            order_summary=order_summary,
            customer_summary=customer_summary,
        )

    @property
    def period_start(self) -> pd.Timestamp:
        """Start time of the period."""
        return self.period.start_time

    @property
    def period_end(self) -> pd.Timestamp:
        """End time of the period."""
        return self.period.end_time

    @property
    def customer_data_long(self) -> pd.DataFrame:
        """Long format of the customer data with period info."""
        df = self.customer_summary.data_long
        df["period_name"] = self.name
        df["period"] = self.period
        return df

    @property
    def order_data_long(self) -> pd.DataFrame:
        """Long format of the order data with period info."""
        df = self.order_summary.data_long
        df["period_name"] = self.name
        df["period"] = self.period
        return df

    def select_customers(self, customer_ids: set[str]) -> "PeriodData":
        """Select customers from all data in this period."""
        return PeriodData(
            name=self.name,
            period=self.period,
            order_summary=self.order_summary.select_customers(customer_ids),
            customer_summary=self.customer_summary.select_customers(customer_ids),
        )

    def drop_customers(self, customer_ids: set[str]) -> "PeriodData":
        """Drop customers from all data in this period."""
        return PeriodData(
            name=self.name,
            period=self.period,
            order_summary=self.order_summary.drop(customer_ids),
            customer_summary=self.customer_summary.drop(customer_ids),
        )

    def drop_otb(self) -> "PeriodData":
        """Drop one-time-buyers (customers with exactly 1 order)."""
        return self.drop_customers(self.customer_summary.customerids_otb)

    def select_otb(self) -> "PeriodData":
        """Select one-time-buyers (customers with exactly 1 order)."""
        return self.select_customers(self.customer_summary.customerids_otb)

    def select_first_orders(self) -> "PeriodData":
        """Select the first order of each customer."""
        return PeriodData.from_ordersummary(
            name=self.name,
            period=self.period,
            order_summary=self.order_summary.select_first_orders(),
        )

    def select_repeat_orders(self) -> "PeriodData":
        """Select all orders after the first order of each customer."""
        return PeriodData.from_ordersummary(
            name=self.name,
            period=self.period,
            order_summary=self.order_summary.select_repeat_orders(),
        )

    def to_deciles(self, splitter: DecileSplitter) -> DecileList:
        """Split the data in this period into deciles."""
        return splitter.split(self.customer_summary)
