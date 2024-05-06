"""Core classes and functions for customer base analysis."""

from typing import Callable, Iterable

# from matplotlib import pyplot as plt
import pandas as pd

from customerbaseanalysis.mixins import (
    CustomerPropertiesMixin,
    OrderPropertiesMixin,
    AccessCustomerSummaryPropertiesMixin,
    AccessOrderSummaryPropertiesMixin,
)

# Could make the classes inherit from a base class or from a mixin that has
# the __getitem__ method to access the data
# def __getitem__(self, cols: str | list[str]) -> pd.DataFrame | pd.Series:
#     # , customer_ids)
#     # make list to always return frame and not series
#     # if isinstance(cols, str):
#     #     cols = [cols]
#     # reset index to be able to read out customer_id
#     # return self.customer_summary.reset_index()[cols]

__all__ = [
    "BasketData",
    "OrderSummary",
    "CustomerSummary",
    "PeriodData",
]


def verify_data(expected_dtypes: dict[str, str], data: pd.DataFrame) -> None:
    """Verify that data has content, all required columns and the expected types."""
    # Verify has rows
    if data.shape[0] == 0:
        raise ValueError("No content in data.")

    # Verify all columns are present
    required_cols = set(expected_dtypes.keys())
    if not required_cols == set(data.columns):
        raise ValueError(
            f"The following columns are required: {required_cols}, "
            f"but got {data.columns}"
        )

    for col, expected_dtype in expected_dtypes.items():
        # Does not work for string types
        # if data[col].dtype != pd.api.types.pandas_dtype(expected_dtype):
        match expected_dtype:
            case "str" | "string" | "object":
                fn = pd.api.types.is_string_dtype
            case "int":
                fn = pd.api.types.is_integer_dtype
            case "float":
                fn = pd.api.types.is_float_dtype
            case "datetime64":
                fn = pd.api.types.is_datetime64_any_dtype
            case _:
                raise ValueError(f"Unknown dtype {expected_dtype}")

        if not fn(data[col]):
            raise ValueError(
                f"Expected {col} to have dtype {expected_dtype}, but got {data[col].dtype}"
            )


class BasketData(OrderPropertiesMixin, CustomerPropertiesMixin):
    """Order data at the item level.

    Attributes:
        basket (pd.DataFrame): Basket data with one item per row. In the future may
            also hold infos such as item count, sku, tax, COGS, etc.
            columns: order_id (str), customer_id (str), timestamp (pd.datetime),
                revenue (float), margin (currency amount, float)
    """

    def __str__(self):
        """Pretty print BasketData object."""
        return (
            f"BasketData\n"
            f"\n"
            f"Items: {self.n_items} \n"
            f"Time: {self.data['timestamp'].min().date()} <> "
            f"{self.data['timestamp'].max().date()}"
            f"Orders: {self.n_orders} \n"
            f"Customers: {self.n_customers} \n"
        )

    @property
    def n_items(self):
        """The number of items in the basket (rows)."""
        return self.data.shape[0]

    def __init__(self, data: pd.DataFrame):
        """Initialize BasketData object after verifying input data."""

        verify_data(
            expected_dtypes={
                "order_id": "str",
                "customer_id": "str",
                "timestamp": "datetime64",
                "revenue": "float",
                "margin": "float",
            },
            data=data,
        )

        # Verify single customer and datetime per order
        order_validation = data.groupby("order_id").agg(
            nunique_customerid=("customer_id", "nunique"),
            nunique_timestamp=("timestamp", "nunique"),
        )
        if not (
            order_validation["nunique_customerid"].eq(1).all()
            and order_validation["nunique_timestamp"].eq(1).all()
        ):
            raise ValueError(
                "Order data with multiple customer_id or timestamp per order_id"
            )

        # fresh numeric index
        self.data = data.copy().reset_index(drop=True)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame):
        """Create BasketData object from a DataFrame after converting data types."""

        # Convert data types
        data["customer_id"] = data["customer_id"].astype(str)
        data["order_id"] = data["order_id"].astype(str)
        data["revenue"] = data["revenue"].astype(float)
        data["margin"] = data["margin"].astype(float)
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        return cls(data)


class OrderSummary(OrderPropertiesMixin, CustomerPropertiesMixin):
    """Order data aggregated at the order_id level.

    Order-level data only. It does not store or generate any data aggregated at the
    customer-level such as number of orders per customer, total revenue per customer,
    etc.

    Attributes:
        data (pd.DataFrame): Order data aggregated at the order_id level.
            columns: order_id (str), customer_id (str), timestamp, revenue,
                margin (currency amount), margin_perc
        _order_num (pd.DataFrame): Running order number for each customer.
            columns: order_id, customer_id, timestamp, order_num
    """

    def __getitem__(self, cols: str | list[str]):
        return self.data[cols]

    def __str__(self):
        """Pretty print OrderSummary object."""
        return (
            f"OrderSummary\n"
            f"\n"
            f"Orders: {self.n_orders} \n"
            f"Customers: {self.n_customers} \n"
            f"Time: {min(self.timestamps)} <> {max(self.timestamps)}"
        )

    def __init__(self, data: pd.DataFrame):

        expected_dtypes = {
            "order_id": "str",
            "customer_id": "str",
            "timestamp": "datetime64",
            "revenue": "float",
            "margin": "float",
            "margin_perc": "float",
        }

        verify_data(expected_dtypes=expected_dtypes, data=data)

        # fresh numeric index
        self.data = data.copy().reset_index(drop=True)
        self._order_num = None

    @classmethod
    def from_basketdata(cls, basket: BasketData) -> "OrderSummary":
        """Aggregate BasketData object at the order_id level to create OrderSummary."""

        # group by customer_id and timestamp to also have them in the resulting table
        order_summary = basket.data.groupby(
            ["order_id", "customer_id", "timestamp"], as_index=False
        ).agg(
            revenue=("revenue", "sum"),
            margin=("margin", "sum"),
        )

        # Calculate margin percentage of each order
        order_summary["margin_perc"] = (
            order_summary["margin"] / order_summary["revenue"]
        )

        return cls(data=order_summary)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame):
        """Create OrderSummary object from a DataFrame after converting data types."""

        # Convert data types
        data["customer_id"] = data["customer_id"].astype(str)
        data["order_id"] = data["order_id"].astype(str)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["revenue"] = data["revenue"].astype(float)
        data["margin"] = data["margin"].astype(float)
        data["margin_perc"] = data["margin_perc"].astype(float)

        return cls(data)

    def data_long(self) -> pd.DataFrame:
        """Long format of the data."""
        return self.data.melt(id_vars=["customer_id", "order_id"], var_name="variable")

    def query(self, query: str) -> "OrderSummary":
        """Select order data by querying."""
        # cannot re-use in select_x() methods because cannot access the variables
        # referenced in the query string
        return OrderSummary(data=self.data.query(query))

    def select_period(self, period: pd.Period | str) -> "OrderSummary":
        """Select all orders within a period.

        The period start_time and end_time are included.
        """
        period = pd.Period(period)

        return OrderSummary(
            data=self.data.query(
                "timestamp >= @period.start_time and timestamp <= @period.end_time"
            )
        )

    def select_customers(self, customer_ids: Iterable[str]) -> "OrderSummary":
        """Select all orders of given customers."""
        # isin() does not allow str
        if isinstance(customer_ids, str):
            raise ValueError("No strings allowed")

        return OrderSummary(data=self.data.query("customer_id.isin(@customer_ids)"))

    def select_until(self, ts: str | pd.Timestamp) -> "OrderSummary":
        """Select all orders until a given timestamp (incl)."""
        ts = pd.to_datetime(ts)

        return OrderSummary(data=self.data.query("timestamp <= @ts"))

    def select_nth(
        self,
        lessthan: int | None = None,
        morethan: int | None = None,
        exactly: int | None = None,
    ) -> "OrderSummary":
        """Select all orders until/from/exactly the n-th order of each customer.

        The n-th order means "n-th position in a customer's order stream". It is
        determined by the order timestamp.
        It does NOT select all orders of the customers having upto, morethan, exactly
        n orders. This would require aggregating by customers.
        
        Args:
            lessthan: All orders until but excl. the `lessthan`-th ( < `lessthan`).
            morethan: All orders after the `morethan`-th order ( > `morethan`).
            exactly: All `exactly`-th orders (== `exactly`).
        """
        # pylint: disable=W0612
        # "Unused variable 'oids'" because used in df.query string

        # Alternative names: select_nth_order
        if sum([lessthan is not None, morethan is not None, exactly is not None]) != 1:
            raise ValueError("Specify exactly one parameter")

        # (Lazy) Calculate running order number for each customer, if not already done
        if self._order_num is None:
            self._order_num = self.data[["order_id", "customer_id", "timestamp"]].copy()
            self._order_num = self._order_num.sort_values("timestamp", ascending=True)
            self._order_num["order_num"] = 1
            self._order_num["order_num"] = self._order_num.groupby("customer_id")[
                "order_num"
            ].cumsum()

        # Order ids of all orders until/from/exactly the n-th order of each customer.
        if lessthan is not None:
            oids = set(self._order_num.query("order_num < @lessthan")["order_id"])
        if morethan is not None:
            oids = set(self._order_num.query("order_num > @morethan")["order_id"])
        if exactly is not None:
            oids = set(self._order_num.query("order_num == @exactly")["order_id"])

        return OrderSummary(data=self.data.query("order_id.isin(@oids)"))

    def select_first_orders(self) -> "OrderSummary":
        """Select the first order of each customer."""
        return self.select_nth(exactly=1)

    def select_repeat_orders(self) -> "OrderSummary":
        """Select all orders after the first order of each customer."""
        return self.select_nth(morethan=1)

    def drop(
        self,
        customer_ids: Iterable[str] | None = None,
        order_ids: Iterable[str] | None = None,
    ) -> "OrderSummary":
        """Drop orders of given customers or order_ids."""

        # Use data to accept customer_ids and order_ids at the same time.
        # Do not have to copy the data because it will be copied in the new object.
        data = self.data

        if customer_ids is not None:
            if isinstance(customer_ids, str):
                raise ValueError("No strings allowed")

            data = data.query("customer_id not in @customer_ids")

        if order_ids is not None:
            data = data.query("order_id not in @order_ids")

        return OrderSummary(data=data)

    def apply_inter_order(
        self, fn: Callable[[pd.DataFrame], pd.DataFrame | pd.Series]
    ) -> pd.DataFrame:
        """Apply a function to each customer's order data sorted by timestamp.

        The order data is first sorted by timestamp and then grouped by customer_id
        before applying the function fn on each group.
        """
        # groupby preserves the order of rows within each group
        return (
            self.data.sort_values(["customer_id", "timestamp"])
            .groupby("customer_id", as_index=True, group_keys=False)
            .apply(fn)
            .reset_index()
        )

    def plot_line(
        self,
        y: str,
        agg_freq: str = "D",
        agg_fn: Callable = sum,
        ylabel: str | None = None,
        **kwargs,
    ) -> "plt.Axes":
        """Plot order data over time, aggregated by freq.

        Args:
            **kwargs: Additional arguments to pass to `seaborn.lineplot`.
        """
        # pylint: disable=C0415
        import seaborn as sns

        df_plot = (
            self.data.groupby(pd.Grouper(key="timestamp", freq=agg_freq))
            .agg({y: agg_fn})
            .reset_index()
        )

        ax = sns.lineplot(data=df_plot, x="timestamp", y=y, **kwargs)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        return ax


class CustomerSummary(CustomerPropertiesMixin):
    """Summary of each customers' orders in a period.

    For each customer, the following statistics are calculated:
        - time_first_order: Timestamp of the first order
        - time_last_order: Timestamp of the last order
        - n_orders: Number of unique orders
        - total_order_revenue: Sum of revenue across all orders
        - mean_order_revenue: Mean revenue across all orders
        - median_order_revenue: Median revenue across all orders
        - total_order_margin: Sum of margin across all orders
        - mean_order_margin: Mean margin across all orders
        - median_order_margin: Median margin across all orders

    Attributes:
        data (pd.DataFrame): Customer summary data with above statistics.
            columns: customer_id (str), <one column for each stat>
    """

    def __str__(self):
        """Pretty print CustomerSummary object."""
        return (
            f"CustomerSummary\n"
            f"\n"
            f"Customers: {self.n_customers}\n"
            f"Orders ({self.n_orders}): {self.time_first_order.date()} <> "
            f"{self.time_last_order.date()}\n"
            f"First Orders: {self.data["time_first_order"].min().date()} <> "
            f"{self.data["time_first_order"].max().date()}\n"
            f"\n"
            f"Total Revenue: {round(self.sum_revenue, 2)}\n"
            f"Total Margin: {round(self.sum_margin, 2)}\n"
        )

    def __init__(self, data: pd.DataFrame):
        """Initialize CustomerSummary object after verifying input data."""

        verify_data(
            expected_dtypes={
                "customer_id": "str",
                "time_first_order": "datetime64",
                "time_last_order": "datetime64",
                "n_orders": "int",
                "total_order_revenue": "float",
                "mean_order_revenue": "float",
                "median_order_revenue": "float",
                "total_order_margin": "float",
                "mean_order_margin": "float",
                "median_order_margin": "float",
            },
            data=data,
        )

        # Verify no duplicates in customer_id
        if data["customer_id"].duplicated().any():
            raise ValueError("data may not contain duplicate customer_ids")

        self.data = data.copy()

    @property
    def data_long(self) -> pd.DataFrame:
        """Long format of the data."""
        return self.data.melt(id_vars="customer_id", var_name="variable")

    @classmethod
    def from_ordersummary(cls, order_summary: OrderSummary) -> "CustomerSummary":
        """Summarize orders of each customer"""

        customer_summary = order_summary.data.groupby(
            "customer_id", as_index=False
        ).agg(
            time_first_order=("timestamp", "min"),
            time_last_order=("timestamp", "max"),
            n_orders=("customer_id", "count"),
            total_order_revenue=("revenue", "sum"),
            mean_order_revenue=("revenue", "mean"),
            median_order_revenue=("revenue", "median"),
            total_order_margin=("margin", "sum"),
            mean_order_margin=("margin", "mean"),
            median_order_margin=("margin", "median"),
        )
        return cls(customer_summary)

    @property
    def n_orders(self) -> int:
        """The total number of orders across all customers."""
        return int(self.data["n_orders"].sum())

    @property
    def sum_revenue(self) -> float:
        """The total revenue across all customers."""
        return float(self.data["total_order_revenue"].sum())

    @property
    def sum_margin(self) -> float:
        """The total margin across all customers."""
        return float(self.data["total_order_margin"].sum())

    @property
    def time_first_order(self) -> pd.Timestamp:
        """Timestamp of the first order."""
        return self.data["time_first_order"].min()

    @property
    def time_last_order(self) -> pd.Timestamp:
        """Timestamp of the last order."""
        return self.data["time_last_order"].max()

    @property
    def aov(self) -> float:
        """Average order value (AOV) across all customers."""
        return float(self.sum_revenue / self.n_orders)

    @property
    def aom(self) -> float:
        """Average order margin (AOM) across all customers."""
        return float(self.sum_margin / self.sum_revenue)

    @property
    def aof(self) -> float:
        """Average order frequency (AOF) across all customers."""
        return float(self.n_orders / self.n_customers)

    @property
    def customerids_otb(self) -> set[str]:
        """Customer ids of one-time-buyers (customers with exactly 1 order)."""
        return self.customerids_numorders(exactly=1)

    def query(self, query: str) -> "CustomerSummary":
        """Select customer data by querying."""
        return CustomerSummary(data=self.data.query(query))

    def select_customers(self, customer_ids: Iterable[str]) -> "CustomerSummary":
        """Select customer data by customer_ids."""
        if isinstance(customer_ids, str):
            raise ValueError("No strings allowed")
        return CustomerSummary(data=self.data.query("customer_id in @customer_ids"))

    def drop(self, customer_ids: Iterable[str]) -> "CustomerSummary":
        """Drop customers by customer_ids."""
        if isinstance(customer_ids, str):
            raise ValueError("No strings allowed")
        return CustomerSummary(data=self.data.query("customer_id not in @customer_ids"))

    def drop_otb(self) -> "CustomerSummary":
        """Drop one-time-buyers (customers with exactly 1 order)."""
        return self.drop(self.customerids_otb)

    def select_otb(self) -> "CustomerSummary":
        """Select one-time-buyers (customers with exactly 1 order)."""
        return self.select_customers(self.customerids_otb)

    def customerids_numorders(
        self,
        lessthan: int | None = None,
        morethan: int | None = None,
        exactly: int | None = None,
    ) -> set[str]:
        """Get ids of customers with a certain number of orders.

        Args:
            lessthan: Customers with stricty less than (<, excl) this number of orders.
            morethan: Customers with strictly more than (>, excl) this number of orders.
            exactly: Customers with exactly this number of orders.
        """
        if sum([lessthan is not None, morethan is not None, exactly is not None]) != 1:
            raise ValueError("Give exactly one argument")

        if lessthan is not None:
            return set(self.data.query("n_orders < @lessthan")["customer_id"])
        if morethan is not None:
            return set(self.data.query("n_orders > @morethan")["customer_id"])
        if exactly is not None:
            return set(self.data.query("n_orders == @exactly")["customer_id"])

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
    ) -> "pd.DataFrame | plt.Axes":
        """Plot a histogram of the data.

        Args:
            x: Column name (str) or function to apply to the data.
            clip_lower: Clip values lower than this value.
            clip_upper: Clip values higher than this value.
            astype: Convert data to this type.
            xlabel: Label for x-axis.
            ylabel: Label for y-axis.
            plot: Whether to plot the data or return it.
            **kwargs: Additional arguments to pass to `seaborne.histplot`.
        """
        # pylint: disable=C0415
        import seaborn as sns

        if isinstance(x, str):
            df_plot = self.data[[x]]
        else:
            df_plot = x(self.data)

        df_plot = df_plot.clip(lower=clip_lower, upper=clip_upper)

        if astype is not None:
            df_plot = df_plot.astype(astype)

        if not plot:
            return df_plot

        x_name = x if isinstance(x, str) else x.columns[0]
        ax = sns.histplot(data=df_plot, x=x_name, **kwargs, ax=ax)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return ax

    def summary(self) -> pd.DataFrame:
        """Summary statistics and decompositions of the customer data."""
        return pd.DataFrame({
                "n_customers": [self.n_customers],
                "n_orders": [self.n_orders],
                "sum_revenue": [self.sum_revenue],
                "sum_margin": [self.sum_margin],
                # profit decomposition
                "aof": [self.aof],
                "aov": [self.aov],
                "aom": [self.aom],
                # per customer
                "avg_revenue_per_customer": [self.sum_revenue / self.n_customers],
                "avg_margin_per_customer": [self.sum_margin / self.n_customers],
                "avg_orders_per_customer": [self.n_orders / self.n_customers],
            }).round(2)


class PeriodData(
    AccessCustomerSummaryPropertiesMixin, AccessOrderSummaryPropertiesMixin
):
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
            f"Period: {self.name}\n"
            f"Time: {self.period.start_time.date()} <> {self.period.end_time.date()}\n"
            f"\n"
            f"Customers: {self.n_customers}\n"
            f"Orders: {self.n_orders}\n"
            f"Total Revenue: {round(self.sum_revenue )}\n"
            f"Total Margin: {round(self.sum_margin)}"
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
