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


def confusion_matrix(y: pd.Series, y_p: pd.Series) -> np.array:
    """Calculates the confusion matrix of a classification model

    Parameters
    ----------
    y : pd.Series
        True values
    y_p : pd.Series
        Predicted values

    Returns
    -------
    np.array
        Confusion Matrix
    """
    cats = y.unique()
    cats = np.sort(cats)
    n = len(cats)
    conf_matr = np.zeros((n,n))

    for i in range(len(y)):
        k = np.where(cats == y.iloc[i])[0][0]
        l = np.where(cats == y_p.iloc[i])[0][0]
        conf_matr[k][l] = conf_matr[k][l] + 1
    return conf_matr


def accuracy_classification(cm: np.array) -> float:
    """Correct predictions vs total number of data points

    Parameters
    ----------
    cm : np.array
        Confusion Matrix

    Returns
    -------
    float
        Accuracy
    """
    return cm.trace()/cm.sum()


def presicion_classification(cm: np.array, i: int = 0) -> float:    
    """True Positives of a given category vs total number of data points

    Parameters
    ----------
    cm : np.array
        Multiclass confusion matrix
    i : int, optional
        Index of category of interests, by default 0

    Returns
    -------
    float
        _description_
    """
    return cm[i][i]/cm[i].sum()


def recall_classification(cm: np.array, i: int = 0) -> float:    
    """True Positives of a given category vs total number of data points

    Parameters
    ----------
    cm : np.array
        Multiclass confusion matrix
    i : int, optional
        Index of category of interests, by default 0

    Returns
    -------
    float
        _description_
    """
    return cm[i][i]/cm[:,i].sum()


def fbeta_classification(cm: np.array, i: int = 0, beta: float = 1.0) -> float:
    p = presicion_classification(cm, i=i)
    r = recall_classification(cm, i=i)
    fb = 0
    den = ((beta**2)*p+r)
    if den !=0:
        fb = (1+beta**2)*p*r/((beta**2)*p+r)
    return fb