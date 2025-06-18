import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import statsmodels.api as sm
from statsmodels.api import GLM, families
from statsmodels.genmod.families.links import Probit

class LogisticReg_with_pvalues:
    def __init__(self, use_auto_weights=True):
        
        self.use_auto_weights = use_auto_weights
        self.sample_weight = None
        self.X_columns = None

    def _compute_class_weights(self, y):
        """Compute sample weights from class imbalance"""
        y_array = np.asarray(y).ravel()
        expected_classes = np.array([0, 1])
        if not np.all(np.isin(expected_classes, y_array)):
            raise ValueError(f"Target variable y must contain both classes 0 and 1. Found: {np.unique(y_array)}")

        class_weights = compute_class_weight(class_weight='balanced', classes=expected_classes, y=y_array)
        weight_map = dict(zip(expected_classes, class_weights))
        sample_weight = np.array([weight_map[label] for label in y_array])
        return sample_weight

    def fit(self, X, y):
        # Ensure X is a DataFrame to preserve column names
        if not isinstance(X, pd.DataFrame):
            self.X_columns = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.X_columns)
        else:
            self.X_columns = list(X.columns)
            
        # 🔽 Convert bool to int to avoid statsmodels casting issues
        X = X.apply(lambda col: col.astype(int) if col.dtype == bool else col)
        # If y is boolean:
        y = y.astype(int)

        if self.use_auto_weights:
            self.sample_weight = self._compute_class_weights(y)

        
        X_sm = sm.add_constant(X)
        self.model = GLM(y, X_sm, family=families.Binomial(link=Probit()), 
                         var_weights=self.sample_weight if self.sample_weight is not None else None)
        self.results = self.model.fit()
        self.feature_names_ = X_sm.columns

        return self

    def summary(self):
        return self.results.summary()

    def coef(self):
        return pd.Series(self.results.params, index=self.feature_names_)

    def pvalues(self):
        return pd.Series(self.results.pvalues, index=self.feature_names_)
    
    def coef_summary(self):
        """Return a DataFrame with coefficients and their p-values."""
        coef_df = pd.DataFrame({
            'Coefficient': self.results.params,
            'P-Value': self.results.pvalues
        })
        coef_df.index.name = 'Variable'
        return coef_df
    
    def predict_proba(self, X):
        """Return class probabilities."""
        if not isinstance(X, pd.DataFrame):
            
            X = pd.DataFrame(X, columns=self.X_columns)

        X = X.apply(lambda col: col.astype(int) if col.dtype == bool else col)    
        X_sm = sm.add_constant(X, has_constant='add')
            
        return self.results.predict(X_sm)  # returns probability of class 1
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

  
        
        
    