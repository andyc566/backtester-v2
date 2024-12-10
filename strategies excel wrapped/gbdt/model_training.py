import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import joblib
from sklearn.model_selection import train_test_split


def train_lgb(X_train, y_train, X_valid, y_valid, w_train, w_valid, num_rounds=500):
    """Train a LightGBM model."""
    model = lgb.LGBMRegressor(n_estimators=num_rounds)
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_valid, y_valid)], eval_sample_weight=[w_valid],
        eval_metric="rmse", early_stopping_rounds=50
    )
    return model


def train_xgb(X_train, y_train, X_valid, y_valid, w_train, w_valid):
    """Train an XGBoost model."""
    model = xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror")
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_valid, y_valid)], sample_weight_eval_set=[w_valid],
        early_stopping_rounds=50, verbose=True
    )
    return model


def train_cbt(X_train, y_train, X_valid, y_valid, w_train, w_valid):
    """Train a CatBoost model."""
    model = cbt.CatBoostRegressor(
        iterations=500, learning_rate=0.05, loss_function="RMSE"
    )
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)])
    return model


def save_model(model, file_path):
    """Save a trained model."""
    joblib.dump(model, file_path)
