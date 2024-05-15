from xgboost import XGBClassifier
from xgboost import XGBRegressor


class XGBoostConfig:
    """
    The hyperparameters used in this classifier are optimized using Optuna
    during experimentation.
    """
    def __init__(self):
        self.best_params = {
            'verbosity': 1,
            'lambda': 0.0012897831631655374,
            'alpha': 0.07295070130798546,
            'max_depth': 3,
            'eta': 0.010996354208650498,
            'gamma': 0.003084118284934084,
            'grow_policy': 'depthwise',
            'learning_rate': 0.034910666476050664,
            'scale_pos_weight': 12,
            'colsample_bytree': 0.9091392953393618,
            'reg_alpha': 4.336889231714178e-08,
            'reg_lambda': 0.008006462256977515,
            'min_child_weight': 4.61829889157942
        }


class CustomXGBoostClassifier:
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.model = XGBClassifier(**self.config.best_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class CustomXGBoostRegressor:
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.model = XGBRegressor(**self.config.best_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
