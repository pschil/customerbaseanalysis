"""Classes for decile analysis."""

import abc
import copy
from typing import Iterable, Callable, Any
import pandas as pd
from customerbaseanalysis.data import CustomerSummary
from customerbaseanalysis.mixins import AccessCustomerSummaryPropertiesMixin
from customerbaseanalysis.foreach import (
    ForeachDF,
    ForeachAsIs,
    verify_single_df_asis,
    ForeachDataElement,
    ForeachIdMapping,
    ForeachKeyMapping,
)

__all__ = [
    "DecileSplitterEqualNumCustomers",
    "DecileSplitterEqualAmount",
    "DecileSplitterCutpoints",
]


class Decile(AccessCustomerSummaryPropertiesMixin):
    """A group of customers which forms a single decile.

    Attributes:
        name: The name of this decile.
        customer_summary: The customers in this decile.
        rank: The position of this decile compared to other deciles. The position in a
            higher-order list should not be stored in the item itself. Here, deciles are
            not reordered and having rank allows more natural comparison with other
            deciles (generated by another splitter) than by name.
    """

    name: str
    customer_summary: CustomerSummary
    rank: int

    def __str__(self) -> str:
        """Pretty print this decile."""
        return (
            f"Single Decile\n"
            f"\n"
            f"Name: {self.name}\n"
            f"Rank: {self.rank}\n"
            f"Num customers: {self.n_customers}\n"
        )

    def __init__(self, name: str, rank: int, customer_summary: CustomerSummary):
        self.name = name
        self.rank = rank
        self.customer_summary = customer_summary

    def select_customers(self, customer_ids: Iterable[str]) -> "Decile":
        """Subset the decile to only include the specified customer ids"""
        return Decile(
            name=self.name,
            rank=self.rank,
            customer_summary=self.customer_summary.select_customers(
                customer_ids=customer_ids
            ),
        )

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for this decile."""
        df = self.customer_summary.summary()
        df.insert(0, "decile_name", self.name)
        df.insert(1, "decile_rank", self.rank)
        return df


class DecileList:
    """Group of 10 deciles ordered by their rank.

    Attributes:
        deciles: List of deciles, ordered from lowest to highest.
        order_col: The column in the customer summary on which the customers where split
            into deciles.
        cutpoints: The cutpoints used to create the deciles.
    """

    deciles: list[Decile]
    order_col: str
    cutpoints: list[float]

    def __str__(self) -> str:
        """Pretty print this decile list."""
        return (
            f"DecileList\n"
            f"\n"
            f"Num deciles: {len(self.deciles)}\n"
            f"Order column: {self.order_col}\n"
            f"Cutpoints: {[round(p,2) for p in self.cutpoints]}\n"
        )

    def __init__(self, deciles: list[Decile], order_col: str, cutpoints: list[float]):

        # Verify there are list elements
        if len(deciles) != 10:
            raise ValueError("deciles must be a list of exactly 10 decile")

        # Verify that deciles are ranked 1 to 10
        if not all(d.rank == i + 1 for i, d in enumerate(deciles)):
            raise ValueError("Deciles must be ordered from 1 to 10")

        # Verify that deciles are ordered from lowest to highest in terms of order_col
        if not all(
            deciles[i].customer_summary.data[order_col].max()
            >= deciles[i + 1].customer_summary.data[order_col].min()
            for i in range(len(deciles) - 1)
        ):
            raise ValueError("Customers in deciles must be ordered wrt to order_col")

        self.deciles = copy.deepcopy(deciles)
        self.cutpoints = copy.deepcopy(cutpoints)
        self.order_col = order_col

    def __iter__(self):
        """Iterator over deciles."""
        return iter(self.deciles)

    def __getitem__(self, key: int | str) -> Decile:
        """Get a decile or a subset of deciles by rank(s) or name(s)."""
        if isinstance(key, int):
            return next(d for d in self if d.rank == key)
        elif isinstance(key, str):
            return next(d for d in self if d.name == key)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def select_customers(self, customer_ids: set[str]) -> "DecileList":
        """Select only the given customer ids across all deciles.

        Subsetting deciles does strictly speaking not make sense because it will not be
        a *Decile*List anymore.
        """
        return DecileList(
            deciles=[d.select_customers(customer_ids) for d in self],
            order_col=self.order_col,
            cutpoints=self.cutpoints,
        )

    @property
    def df_deciles(self) -> pd.DataFrame:
        """Decile membership (decile name and ranke) for every customer."""
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "customer_id": list(d.customer_ids),
                        "decile_name": d.name,
                        "decile_rank": d.rank,
                    }
                )
                for d in self
            ]
        ).reset_index(drop=True)

    @property
    def customer_ids(self) -> set[str]:
        """Get the customer ids across all deciles."""
        return set.union(*[d.customer_ids for d in self])

    @property
    def n_customers(self) -> int:
        """Get the number of customers across all deciles."""
        return sum(d.n_customers for d in self)

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for each decile in a single pd.DataFrame."""

        df = pd.concat([d.summary() for d in self]).reset_index(drop=True)

        def to_percent(s: pd.Series):
            return round(s / s.sum() * 100, 2)

        # stats across deciles
        df["perc_of_customers"] = to_percent(df["n_customers"])
        df["perc_of_orders"] = to_percent(df["n_orders"])
        df["perc_of_revenue"] = to_percent(df["sum_revenue"])
        df["perc_of_margin"] = to_percent(df["sum_margin"])

        # Return in correct col order
        return df[
            [
                "decile_name",
                "decile_rank",
                "n_customers",
                "perc_of_customers",
                "n_orders",
                "perc_of_orders",
                "sum_revenue",
                "perc_of_revenue",
                "sum_margin",
                "perc_of_margin",
                "aof",
                "aov",
                "aom",
                "avg_revenue_per_customer",
                "avg_margin_per_customer",
                "avg_orders_per_customer",
            ]
        ]

    def foreach_decile(
        self,
        df: Callable[[Decile, Any], pd.DataFrame | dict] | None = None,
        asis: Callable[[Decile, Any], Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, Any]:
        """Apply a function `df` or `asis` to each decile."""

        verify_single_df_asis(df=df, asis=asis)

        def get_deciles():
            for d in self.deciles:
                yield ForeachDataElement(data=d, metainfo=None)

        if df is not None:
            foreach = ForeachDF(
                fn_df=df,
                id_mappings=[
                    ForeachIdMapping(name="decile_n", fn_id=lambda i, _: i),
                    ForeachIdMapping(
                        fn_id=lambda _, fde: fde.data.rank, name="decile_rank"
                    ),
                    ForeachIdMapping(
                        fn_id=lambda _, fde: fde.data.name, name="decile_name"
                    ),
                ],
            )
        else:
            foreach = ForeachAsIs(
                fn_asis=asis,
                key_mapping=ForeachKeyMapping(fn_key=lambda _, fde: fde.data.name),
            )

        return foreach(get_deciles(), **kwargs)


class DecileSplitter(abc.ABC):
    """Base class for decile splitters.

    Attributes:
        order_col: The column in customer_summary to use for splitting.
    """

    def __init__(self, order_col: str) -> None:
        self.order_col = order_col

    def _revert_deciles(self, ser: pd.Series):
        return abs(ser - 10)

    @abc.abstractmethod
    def _assign_customers_to_deciles(
        self, customer_summary: CustomerSummary
    ) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError

    def split(self, customer_summary: CustomerSummary) -> DecileList:
        """Split a customer summary into deciles based on a given order column.

        Args:
            customer_summary: The customer summary to split.
        """
        if self.order_col not in customer_summary.data.columns:
            raise ValueError(
                f"Column '{self.order_col}' not present in customer_summary"
            )

        df_cs, cutpoints = self._assign_customers_to_deciles(
            customer_summary=customer_summary,
        )
        cutpoints = list(cutpoints)

        deciles = DecileList(
            [
                Decile(
                    name=f"Decile {n}",
                    rank=n,
                    customer_summary=customer_summary.select_customers(
                        customer_ids=set(df_d["customer_id"])
                    ),
                )
                for n, df_d in df_cs.groupby("decile")
            ],
            cutpoints=cutpoints,
            order_col=self.order_col,
        )
        return deciles


class DecileSplitterCutpoints(DecileSplitter):
    """Split customers into 10 deciles, with cutpoints defined by the user.

    Attributes:
        cutpoints: The cutpoints at which order_col is split.
    """

    def __str__(self) -> str:
        """Pretty print this decile splitter."""
        return (
            f"DecileSplitterCutpoints\n"
            f"\n"
            f"Split to customers at given cutpoints into deciles\n"
            f"order_col: {self.order_col}\n"
            f"cutpoints: {self.cutpoints}\n"
        )

    def __init__(self, order_col: str, cutpoints: Iterable[float]):
        super().__init__(order_col=order_col)
        self.cutpoints = cutpoints

    def _assign_customers_to_deciles(
        self, customer_summary: CustomerSummary
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_cs = customer_summary.data.copy()

        # TODO: Does this need to be sorted?
        # df_cs = df_cs.sort_values(self.order_col, ascending=True)

        df_cs["decile"], cutpoints = pd.cut(
            x=df_cs[self.order_col],
            bins=self.cutpoints,
            labels=False,
            include_lowest=True,
            retbins=True,
        )
        df_cs["decile"] = self._revert_deciles(df_cs["decile"])
        return df_cs, cutpoints


class DecileSplitterEqualNumCustomers(DecileSplitter):
    """Split customers into 10 deciles, each with approx equal number of customers."""

    def __str__(self) -> str:
        """Pretty print this decile splitter."""
        return (
            f"DecileSplitterEqualNumCustomers\n"
            f"\n"
            f"Split to equal num customers (sorted by order_col) in each decile \n"
            f"order_col: {self.order_col}\n"
        )

    def _assign_customers_to_deciles(
        self, customer_summary: CustomerSummary
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_cs = customer_summary.data.copy()

        df_cs["decile"], cutpoints = pd.qcut(
            df_cs[self.order_col], q=10, labels=False, retbins=True
        )
        df_cs["decile"] = self._revert_deciles(df_cs["decile"])
        return df_cs, cutpoints


class DecileSplitterEqualAmount(DecileSplitter):
    """Split customers into 10 deciles, each with approx equal sum(`order_col`)."""

    def __str__(self) -> str:
        """Pretty print this decile splitter."""
        return (
            f"DecileSplitterEqualAmount\n"
            f"\n"
            f"Split to equal amount of order_col in each decile \n"
            f"order_col: {self.order_col}\n"
        )

    def _assign_customers_to_deciles(
        self,
        customer_summary: CustomerSummary,
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_cs = customer_summary.data.copy()

        df_cs = df_cs.sort_values(self.order_col, ascending=True)

        df_cs["decile"], cutpoints = pd.cut(
            df_cs[self.order_col].cumsum(), bins=10, labels=False, retbins=True
        )
        df_cs["decile"] = self._revert_deciles(df_cs["decile"])
        return df_cs, cutpoints
