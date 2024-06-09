"""Mixins to use in other classes."""

from typing import Callable
import pandas as pd
from matplotlib import pyplot as plt


class OrderPropertiesMixin:
    """Properties of order data calculated from `self.data`."""

    @property
    def timestamps(self) -> set[pd.Timestamp]:
        """The timestamps in the object."""
        return set(self.data["timestamp"])

    @property
    def time_first_order(self) -> pd.Timestamp:
        """Timestamp of the first order."""
        return self.data["timestamp"].min()

    @property
    def time_last_order(self) -> pd.Timestamp:
        """Timestamp of the last order."""
        return self.data["timestamp"].max()

    @property
    def n_orders(self) -> int:
        """The number of orders (unique order_ids) in the object."""
        return int(self.data["order_id"].nunique())

    @property
    def order_ids(self) -> set[str]:
        """The order_ids in the object"""
        return set(self.data["order_id"])

    @property
    def sum_revenue(self) -> float:
        """The total revenue in the object."""
        return float(self.data["revenue"].sum())

    @property
    def sum_profit(self) -> float:
        """The total profit in the object."""
        return float(self.data["profit"].sum())


class CustomerPropertiesMixin:
    """Properties of customer data calculated from `self.data`."""

    @property
    def n_customers(self) -> int:
        """The number of customers (unique customer ids) in the object."""
        return int(self.data["customer_id"].nunique())

    @property
    def customer_ids(self) -> set[str]:
        """The customer_ids in the object."""
        return set(self.data["customer_id"])


class AccessOrderSummaryPropertiesMixin:
    """Properties of self.order_summary."""

    @property
    def time_first_order(self) -> pd.Timestamp:
        """The timestamp of the first order."""
        return self.order_summary.time_first_order

    @property
    def time_last_order(self) -> pd.Timestamp:
        """The timestamp of the last order."""
        return self.order_summary.time_last_order

    @property
    def n_orders(self) -> int:
        """The number of unique orders."""
        return self.order_summary.n_orders

    @property
    def order_ids(self) -> set[str]:
        """The unique order_ids."""
        return self.order_summary.order_ids


class AccessCustomerSummaryMixin:
    """Properties of self.customer_summary."""

    @property
    def n_customers(self) -> int:
        """The number of customers (unique customer ids)."""
        return self.customer_summary.n_customers

    @property
    def customer_ids(self) -> set[str]:
        """All customer_ids."""
        return self.customer_summary.customer_ids

    @property
    def sum_revenue(self) -> float:
        """Total revenue of all orders."""
        return self.customer_summary.sum_revenue

    @property
    def sum_profit(self) -> float:
        """Total profit of all orders."""
        return self.customer_summary.sum_profit
    
    @property
    def avg_revenue(self) -> float:
        """Average revenue per customer."""
        return self.customer_summary.avg_revenue
    
    @property
    def avg_profit(self) -> float:
        """Average profit per customer."""
        return self.customer_summary.avg_profit

    @property
    def aof(self) -> float:
        """Average order frequency."""
        return self.customer_summary.aof

    @property
    def aov(self) -> float:
        """Average order value."""
        return self.customer_summary.aov

    @property
    def aom(self) -> float:
        """Average order revenue."""
        return self.customer_summary.aom

    def plot_hist(
        self,
        x: str | Callable[[pd.DataFrame], pd.DataFrame],
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        astype: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        plot: bool = True,
        ax=None,
        **kwargs,
    ) -> pd.DataFrame | plt.Axes:
        """See `CustomerSummary.plot_hist`."""

        return self.customer_summary.plot_hist(
            x=x,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            astype=astype,
            xlabel=xlabel,
            ylabel=ylabel,
            plot=plot,
            ax=ax,
            **kwargs,
        )
