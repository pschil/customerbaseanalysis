from importlib.resources import files
import modin.pandas as pd


def _load_dataset(filename: str) -> pd.DataFrame:
    """Load a dataset from the data folder."""
    # csv goes into joinpath and not files() because not a proper package (no __init__.py)
    return pd.read_csv(files("customerbaseanalysis.datasets").joinpath("csv", filename))


def load_cdnow_full() -> pd.DataFrame:
    """CDNOW Full dataset

    The entire purchase history up to the end of June 1998 of the cohort of 23570
    customers who made their first-ever purchase at CDNOW in the first quarter of 1997.
    The original used by Fader and Hardie (2001).
    """
    df = _load_dataset("CDNOW_master_modified.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["customer_id"] = df["customer_id"].astype(str)
    return df
