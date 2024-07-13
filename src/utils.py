import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import pickle
from exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as error:
        raise CustomException(error, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        results = {}

        for model_name in range(len(list(models))):

            model = list(models.values())[model_name]

            model.fit(X_train, y_train)


            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            results[list(models.keys())[model_name]] = test_score
        return results

    except Exception as e:
        raise CustomException(e, sys)