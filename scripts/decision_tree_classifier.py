import pandas as pd
import numpy as np
from typing import List, Dict
import math

class Decision_Tree_Classifier:
    def __init__(self) -> None:
        self.res_dict = {}
    
    def fit(self, X, y, X_cat_cols):
        self.res_dict = self.des_tree_nodes(X, y, X_cat_cols)
        
    def predict(self, X):
        return X.apply(lambda x: self.predict_single(x, self.res_dict), axis=1)

    def predict_single(self, x, res_dict):
        tp = res_dict['type']
        attr = res_dict['attr']
        if tp == 'cat':
            res = res_dict['res'][x[attr]]
        else:
            val = res_dict['vals']
            i = self.recover_index(x[attr],val)
            res = res_dict['res'][str(i)]
        
        if isinstance(res,dict):
            return self.predict_single(x, res)
        else:
            return res

    def new_nodes(self, X: pd.DataFrame, y: pd.Series, attr_max_gain: str, X_cat_cols) -> Dict:
        dct_res = {}
        cols = list(X.columns)
        cols.remove(attr_max_gain)
        X_upd = X[cols]
        Xcc = X_cat_cols.copy()
        if attr_max_gain in X_cat_cols:
            Xcc.remove(attr_max_gain)
        
        dct_res['attr'] = attr_max_gain
        if attr_max_gain in X_cat_cols:
            dct_res['type'] = 'cat'
            dct_res['vals'] = np.sort(X[attr_max_gain].unique())
            childs = {}

            for u in dct_res['vals']:
                Xn = X_upd.loc[X[attr_max_gain]==u]
                yn = y.loc[X[attr_max_gain]==u]
                if len(yn)>0:
                    childs[u] = yn.iloc[0]
                    if len(yn.unique()) > 1:
                        childs[u] = self.des_tree_nodes(Xn, yn, Xcc)
            dct_res['res'] = childs
        else:
            dct_res['type'] = 'cont'
            splits = self.get_splits(X[attr_max_gain], y)
            dct_res['vals'] = splits
            childs = {}
            for i in range(len(splits) + 1):                
                Xn = self.iloc_ranges(X_upd, X[attr_max_gain], splits, i)
                yn = self.iloc_ranges(y, X[attr_max_gain], splits, i)
                if len(yn)>0:
                    childs[str(i)] = self.des_tree_nodes(Xn, yn, Xcc)
            dct_res['res'] = childs
        return dct_res
    
    def des_tree_nodes(self, X: pd.DataFrame, y: pd.Series, X_cat_cols):
        cols = list(X.columns)
        attr_max_gain = self.get_max_ig(X, y, X_cat_cols)
        cols.remove(attr_max_gain)
        if len(cols)!=0:
            out = self.new_nodes(X, y, attr_max_gain, X_cat_cols)
            return out
        y_count = {}
        for u in y.unique():
            y_count[u] = y.loc[y==u].count()
        out = max(y_count, key=y_count.get)
        return out
    
    def info_gain_cat(self, x:pd.Series, y:pd.Series) -> float:
        """Calculates the info gain of using a given categorical attribute to describe categorical data

        Parameters
        ----------
        x : pd.Series
            Categorical values of the attribute
        y : pd.Series
            Categorical target

        Returns
        -------
        float
            Information gain
        """
        ES = self.entropy(y)
        ESv = 0
        nt = len(y)
        if nt != 0:
            for u in np.sort(x.unique()):
                Sv = y.loc[x==u]
                n = len(Sv)        
                ESv += - (n/nt)*self.entropy(Sv)
        return ES - ESv

    def info_gain_con(self, x:pd.Series, y:pd.Series) -> float:
        """Calculates the info gain of using a given continuous attribute to describe categorical data

        Parameters
        ----------
        x : pd.Series
            Categorical values of the attribute
        y : pd.Series
            Categorical target

        Returns
        -------
        float
            Information gain
        """
        splits = self.get_splits(x, y)
        ES = self.entropy(y)
        ESv = 0
        nt = len(y)
        if nt != 0:
            for i in range(len(splits) + 1):
                Sv = self.iloc_ranges(y, x, splits, i)
                n = len(Sv)
                ESv += - (n/nt)*self.entropy(Sv)
        return ES - ESv

    def get_max_ig(self, X: pd.DataFrame, y: pd.Series, X_cat_cols: List) -> str:
        """Identifies the column of X with the maximum info gain related to the target y

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        y : pd.Series
            Target data
        X_cat_cols : List
            List of categorical columns of X

        Returns
        -------
        str
            Name of the column with maximum information gain
        """
        dct = {}
        cols = list(X.columns)
        for attr in cols:
            if attr in X_cat_cols:
                dct[attr] = self.info_gain_cat(X[attr], y)
            else:
                dct[attr] = self.info_gain_con(X[attr], y)
        return max(dct, key=dct.get)
    
    def iloc_ranges(self, y: pd.Series, x: pd.Series, splits: List, i: int) -> pd.Series:
        """Filters the data from the series y for which the continuous data x is between the values split[i-1] and split[i]

        Parameters
        ----------
        x : pd.Series
            Continuous data to use as base for the filtering
        y : pd.Series or pd.DataFrame
            Data to be filtered
        splits : List
            Values at which the values of x can be splitted
        i : int
            Index of the split to be used


        Returns
        -------
        pd.Series
            Filtered data
        """
        if len(x) > 0:
            if i > 0 and i < len(splits):
                Sv = y.loc[(x>splits[i-1]) & (x<=splits[i])]
            elif i == 0:
                Sv = y.loc[x<=splits[0]]
            else:
                Sv = y.loc[x>splits[-1]]
            return Sv
        return pd.Series([])

    def recover_index(self, val: float, splits: List) -> int:
        """Recovers the index at which val falls whitin the range of the values specified by splits

        Parameters
        ----------
        val : pd.Series
            Value to evaluate
        splits : List
            Limits of the split ranges.

        Returns
        -------
        int
            Recovered index
        """
        for i,v in enumerate(splits):
            if i > 0:
                if val > splits[i-1] and val <= v:
                    return i
            else:
                if val <= v:
                    return i
        return len(splits)
    
    def get_splits(self, x:pd.Series, y:pd.Series) -> np.array:
        """Returns the splits of the categorical data x, using the median x for each category of y

        Parameters
        ----------
        x : pd.Series
            Input data for which the splits will be calculated
        y : pd.Series
            Target categorical data 

        Returns
        -------
        np.array
            Limits of the splits
        """
        vals = []
        for u in y.unique():
            x_np = x.loc[y==u].to_numpy()
            vals.append(np.median(x_np))
        vals_sorted = np.sort(np.unique(np.array(vals)))
        return vals_sorted
    
    def entropy(self, y: pd.Series) -> float:
        """Calculates the entropy of y

        Parameters
        ----------
        y : pd.Series
            Categorical value for which the entropy will be calculated

        Returns
        -------
        float
            Entropy
        """
        entropy = 0
        ntt = len(y)
        if ntt != 0:
            for u in y.unique():
                y_c = y.loc[y==u]
                p = len(y_c)/ntt
                if p != 0:
                    entropy += -p*math.log2(p)
        return entropy