import pandas as pd
import numpy as np
from typing import Tuple

from scipy.optimize import minimize

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()) + '/../scripts/')
import utils as utl

class LogisticRegression:
    def __init__(self) -> None:
        self.w = None
        self.y_res = None

    def fit(self, X: pd.DataFrame, y: pd.Series, bias: float = 0, w: np.array = None):
        """fit X data towards y target using multiclass logistic regression

        Parameters
        ----------
        X : pd.DataFrame
            Independent data
        y : pd.Series
            Target data
        bias : float, optional
            Constant vertical shift, by default 0
        w : np.array, optional
            Initial guess of the weights, by default all ones                     
        """              
        if w is None:
            w = np.ones(X.shape[1]+1)

        y_res = pd.DataFrame()
        for cat in np.sort(y.unique()):
            y_np = y.copy().to_numpy()
            y_np[y_np!=cat]=-1
            y_np[y_np==cat]=1
            y_np[y_np==-1]=0
            y_res[cat] = y_np

        X_mod = LogisticRegression.add_bias_column(X)
        X_np = X_mod.to_numpy()

        # Inicializes data
        J = np.ones(len(y_res.columns))
        w_df = pd.DataFrame()
        for i,val in enumerate(y_res.columns):
            w_df[val] = 0.1*np.ones(X.shape[1]+1)
        w = w_df.to_numpy()
        y_c = y_res.to_numpy()

        # Minimizes the cost function
        for i,val in enumerate(y_res.columns):
            res = minimize(LogisticRegression.lcf_for_min, w[:,i], args=(X_np, y_c[:,i]))
            w[:,i] = res.x        
        self.w = w
        self.y_res = y_res

    def predict(self, X: np.array) -> np.array:
        """Predicts resulting categories based on input data X

        Parameters
        ----------
        X : np.array
            Input data

        Returns
        -------
        np.array
            Predicted categories
        """
        X_mod = LogisticRegression.add_bias_column(X)
        X_np = X_mod.to_numpy()

        # Predictions
        y_pred = pd.DataFrame()

        for i,val in enumerate(self.y_res.columns):
            y_pred[val] = LogisticRegression.log_reg_predict(X_np, self.w[:,i])
            #y_pred[val] = utl.min_max_scaling(y_pred[val])[0]

        orig_cols = y_pred.columns
        y_pred['res'] = y_pred.apply(lambda x: np.argmax(x[orig_cols].to_numpy()) ,axis =1)
        print(y_pred)
        return y_pred['res']

    @staticmethod
    def log_reg_predict(X: np.array, w: np.array) -> np.array:
        """Single class prediction using a sigmoid of the linear combination of X and w

        Parameters
        ----------
        X : np.array
            Input data with bias column of ones
        w : np.array
            Weights

        Returns
        -------
        np.array
            resulting predictions
        """
        y_pred = (w*X).sum(axis=1)
        den = 1 + np.exp(-y_pred)
        return 1/den

    @staticmethod
    def add_bias_column(X: pd.DataFrame) -> pd.DataFrame:
        """Adds the column 'bias_col' of ones for the bias calculation

        Parameters
        ----------
        X : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Input data plus bias column
        """
        X_mod = X.copy()
        ones_aux = np.ones(X.shape[0])
        X_mod['bias_col'] = ones_aux
        cols = X_mod.columns
        cols_reordered = [cols[i-1] for i,x in enumerate(cols)]
        X_mod = X_mod[cols_reordered]
        return X_mod

    @staticmethod
    def log_cost_function(w: np.array, X: np.array, y: np.array) -> float:
        """Calculates the log cost function for binary logistic regression

        Parameters
        ----------
        w : np.array
            Weights
        X : np.array
            Input values
        y : np.array
            Expected outputs

        Returns
        -------
        float
            Logistic cost function result
        """
        h = LogisticRegression.log_reg_predict(X, w)
        return - (1/len(X))*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    @staticmethod
    def lcf_for_min(w: np.array, *args: Tuple[np.array, np.array]) -> float:
        """Calculates the log cost function for binary logistic regression in format necessary for minimization algorithm

        Parameters
        ----------
        w : np.array
            _description_

        **args: Tuple[np.array, np.array]
            np.array: Input values X
            np.array: Expected outputs y
        Returns
        -------
        float
            Logistic cost function result
        """
        X = args[0]
        y = args[1]
        return LogisticRegression.log_cost_function(w, X, y)