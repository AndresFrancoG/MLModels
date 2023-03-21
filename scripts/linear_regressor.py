import pandas as pd
import numpy as np
from typing import Tuple

from scipy.optimize import minimize


def sum_of_squares_error(y: pd.Series,w: np.array, ph: pd.DataFrame, regularized: bool = True, lamb: float = 0.1) -> float:
    """Calculates the sum of squares error function

    Parameters
    ----------
    y : pd.Series
        target variable
    w : np.array
        current weights
    ph : pd.DataFrame
        phi function evaluated for every entry of X
    regularized : bool, optional
        Includes weight decay regularization to avoid overfitting, by default True
    lamb : float, optional
        Regularization parameter, by default 0.1
        
    Returns
    -------
    float
        Sum of squared error
    """
    res = 0
    wt = np.array([w])
    ph_np = ph.to_numpy()
    for i in range(len(y)):
        val = y.iloc[i]
        ph_i = ph_np[i].reshape(-1,1)
        res = res + (0.5*(val - np.matmul(wt,ph_i))**2)[0][0]
        if regularized:
            res = res + (0.5*lamb*np.matmul(wt,w))[0] 
    return res

def sose_for_min(w :np.array,*args: Tuple[pd.Series, pd.DataFrame, bool, float]) -> float:
    """Calls the funcion sum_of_squares_error in the format apropriate for minimization with scipy

    Parameters
    ----------
    w : np.array
        Initial weigths
    *args: Tuple[pd.Series, pd.DataFrame, bool, float]
        pd.Series: target data
        pd.DataFrame: phi functions evaluated at X
        bool: Include weight decay regularization. Default 
        float: value of we
        
    Returns
    -------
    float
        Sum of squares error
    """
    y = args[0]
    ph = args[1]
    if len(args) > 2:
        regularized = args[2]
    else:
        regularized = True
    
    if len(args) > 3:
        lamb = args[3]
    else:
        lamb = 0.1
    return sum_of_squares_error(y, w, ph, regularized=regularized, lamb=lamb)

class LinearRegression:
    def __init__(self) -> None:
        self.w = None
        self.ph = None

    def fit(self, X: pd.DataFrame, y: pd.Series, bias: float = 0, basis_function: str  = 'lineal', w: np.array = None,
            regularized: bool = True, lamb: float = 0.1):
        """fit X data towards y target using Linear regression

        Parameters
        ----------
        X : pd.DataFrame
            Independent data
        y : pd.Series
            Target data
        bias : float, optional
            Constant vertical shift, by default 0
        basis_function : str, optional
            Basis functions Phi of the linear regressor, by default 'lineal'
                'lineal': Phi = X
                'sigmoid': Phi = Logistic Sigmoidal function
                other values: Phi = 1
        w : np.array, optional
            Initial guess of the weights, by default all ones
        regularized : bool, optional
            Includes weight decay regularization to avoid overfitting, by default True
        lamb : float, optional
            Regularization parameter, by default 0.1                       
        """
        if w is None:
            w = np.ones(X.shape[1]+1)
        match basis_function:
            case 'lineal':
                self.ph = X
            case 'sigmoid':
                X_mean = X.mean(axis=0)
                a = (X - X_mean).to_numpy()
                ph_np = 1/(1 + np.exp(-a))
                self.ph = pd.DataFrame(ph_np, columns=X.columns)
            case 'gaussian':
                X_mean = X.mean(axis=0)
                a = (X - X_mean).to_numpy()
                ph_np = np.exp(-np.power(a,2))
                self.ph = pd.DataFrame(ph_np, columns=X.columns)
            case 'tanh':
                X_mean = X.mean(axis=0)
                a = (X - X_mean).to_numpy()
                ph_np = np.tanh(a)
                self.ph = pd.DataFrame(ph_np, columns=X.columns)                
            case _:
                self.ph = pd.DataFrame(np.ones(X.shape), columns=X.columns)

        ones_aux = np.ones(X.shape[0])
        self.ph['bias_col'] = ones_aux
        cols = self.ph.columns
        cols_reordered = [cols[i-1] for i,x in enumerate(cols)]
        self.ph = self.ph[cols_reordered]
        res = minimize(sose_for_min, w, args=(y, self.ph, regularized, lamb))
        self.w = res.x

    def predict(self, X: pd.DataFrame):
        y_trained = self.ph.apply(lambda x: np.nansum(x.to_numpy()*self.w),axis = 1)
        return y_trained