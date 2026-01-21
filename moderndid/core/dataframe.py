"""DataFrame compatibility layer for pandas/polars interoperability."""

import pandas as pd
import polars as pl

DataFrame = pd.DataFrame | pl.DataFrame


def to_polars(df: DataFrame) -> pl.DataFrame:
    """Convert a DataFrame to polars.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (pandas or polars).

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def to_pandas(df: DataFrame) -> pd.DataFrame:
    """Convert a DataFrame to pandas.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (pandas or polars).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


def is_polars(df: DataFrame) -> bool:
    """Check if DataFrame is a polars DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    bool
        True if polars DataFrame, False otherwise.
    """
    return isinstance(df, pl.DataFrame)


def is_pandas(df: DataFrame) -> bool:
    """Check if DataFrame is a pandas DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    bool
        True if pandas DataFrame, False otherwise.
    """
    return isinstance(df, pd.DataFrame)
