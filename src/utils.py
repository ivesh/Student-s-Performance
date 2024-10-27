import os,sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}
        model_keys = list(models.keys())
        model_values = list(models.values())        

        for i in range(len(model_keys)):
            model = model_values[i]
            model_name = model_keys[i]

            # Perform hyperparameter tuning if parameters are specified for the model
            if model_name in params and params[model_name]:
                param_grid = params[model_name]
                # Use GridSearchCV for exhaustive search, or RandomizedSearchCV for random search
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            # Train and evaluate the model
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        raise CustomException(e,sys)