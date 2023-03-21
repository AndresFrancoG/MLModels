import pandas as pd
import numpy as np
from typing import Tuple


def min_max_scaling(ds: pd.Series) -> Tuple[pd.Series, Tuple[float,float]]:
    """Linearly scales a pandas series from min -> max to 0 -> 1

    Parameters
    ----------
    ds : pd.Series
        Series to scale

    Returns
    -------
    Tuple[pd.Series, Tuple[float,float]]
        pd.Series: Scaled Series
        Tuple[float,float]
            float: Minimum of the original series
            float: Maximum of the original series
    """
    mx = np.nanmax(ds)
    mn = np.nanmin(ds)
    delta = mx - mn
    ds_out = (ds - mn)/delta
    return ds_out, (mn,mx)


def r_squared(y_orig: np.array, y_pred: np.array):
    """Calculates the R squared of a target and predicted arrays

    Parameters
    ----------
    y_orig : np.array
        Target array
    y_pred : np.array
        Predicted array

    Returns
    -------
    _type_
        R squared
    """
    ssreg = np.nansum(np.power(y_orig - y_pred,2))
    sstot = np.nansum(np.power(y_orig - np.nanmean(y_pred),2))
    Rsqrd = 1 - ssreg/sstot
    return Rsqrd