"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.2
"""
import numpy as np 
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import Tuple

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    data['FlashStorage'] = data['FlashStorage'].fillna(0)
    data['MemorySizeHDD_TB'] = data['MemorySizeHDD_TB'].fillna(0)
    data['MemorySizeHDD_GB'] = data['MemorySizeHDD_GB'].fillna(0)
    data['MemorySizeSSD'] = data['MemorySizeSSD'].fillna(0)

    X = data.drop('Price', axis=1)
    y = data['Price']

    return X, y

def run_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae

def optimize_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestRegressor(random_state=0)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_predictions = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, best_predictions)

    return best_model, best_mae

def run_price_prediction_model(data: pd.DataFrame) -> None:
    X, y = preprocess_data(data)
    model, mae = run_model(X, y)
    best_model, best_mae = optimize_model(X, y)
    print(f"Initial MAE: {mae}, Optimized MAE: {best_mae}")